from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.configs.config import CONF
from utils.utils_fn import debug
from transformers import DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer
from models.lang_bert_module.bert.xbert import BertConfig, BertForMaskedLM
from models.lang_bert_module.bert.tokenization_bert import BertTokenizer
from models.transformer.attention import MultiHeadAttention, PredictionHead, PositionEmbeddingHead


class LangBertModule(nn.Module):
    def __init__(self, num_class, use_lang_classifier=True, pc_hidden_size=128, mlm_hidden_size=128,
                 lang_hidden_size=128, head=4, masked_ratio=0.15, depth=2, unfreeze=6, use_distil=False):
        super().__init__()
        self.use_lang_classifier = use_lang_classifier
        self.num_classes = num_class
        self.lang_hidden_size = lang_hidden_size
        self.pc_hidden_size = pc_hidden_size
        self.mlm_hidden_size = mlm_hidden_size
        self.unfreeze = unfreeze
        self.use_distil = use_distil
        self.depth = depth

        # bert module
        if self.use_distil:
            self.bert_config = DistilBertConfig()
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                CONF.DISTILBERT_TEXT_ENCODER)
            self.text_encoder = DistilBertForMaskedLM.from_pretrained(
                CONF.DISTILBERT_TEXT_ENCODER, config=self.bert_config)
        else:
            self.bert_config = BertConfig.from_json_file(CONF.BERT_CONFIG)
            self.tokenizer = BertTokenizer.from_pretrained(
                CONF.BERT_TEXT_ENCODER)
            self.text_encoder = BertForMaskedLM.from_pretrained(
                CONF.BERT_TEXT_ENCODER, config=self.bert_config)

        # project layer
        text_width = self.text_encoder.config.hidden_size
        self.proj = nn.Linear(text_width, lang_hidden_size)

        self.lang_cls = nn.Sequential(
            nn.Linear(self.lang_hidden_size, self.num_classes),
            nn.Dropout()
        )

        self.mask_ratio = masked_ratio
        self.pc_proj = nn.Sequential(
            nn.Linear(self.pc_hidden_size, self.mlm_hidden_size),
            nn.GELU()
        )

        # mlm decoder
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=self.mlm_hidden_size, d_k=self.mlm_hidden_size // head, d_v=self.mlm_hidden_size // head, h=head) for i in range(depth))

        # self attention for dist weight
        self.dist_fc = nn.ModuleList(
            nn.Sequential(  # 4 128 256 4(head)
                nn.Linear(4, 32),  # xyz, dist
                nn.ReLU(),
                nn.LayerNorm(32),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
                nn.Linear(32, 4)
            ) for i in range(depth))

        # use whole box to generate pos embedding
        self.pos_head = nn.ModuleList(
            PositionEmbeddingHead(input_channel=6) for i in range(depth))

        # auxiliary dist weight matrix
        self.use_dist_weight_matrix = True

        # prediction heads
        self.prediction = PredictionHead(hidden_size=self.mlm_hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init()

    def init(self):
        self.hidden_layers = 6 if self.use_distil else 12
        assert(self.unfreeze <= self.hidden_layers)
        unfreeze_list = [
            "layer."+str(l) for l in range(self.hidden_layers-self.unfreeze, self.hidden_layers)]
        unfreeze_list += ["bert.pooler", "cls.", "vocab_"]
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
            # for ele in unfreeze_list:
            #     if ele in name:
            #         param.requires_grad = True
            #         break

    # encode text embedding
    def encode(self, input_ids, attention_mask):
        if self.use_distil:
            text_output = self.text_encoder.distilbert(
                input_ids, attention_mask=attention_mask, return_dict=True)
        else:
            text_output = self.text_encoder.bert(input_ids, attention_mask=attention_mask,
                                                 return_dict=True, mode='text')
        # encode
        lang_fea = text_output.last_hidden_state

        # proj
        lang_fea = self.proj(lang_fea)
        return lang_fea

    def forward(self, data_dict):
        """
            encode the input descriptions
        """
        input_ids = data_dict['input_ids']
        bert_attention_mask = data_dict['bert_attention_mask']

        # reshape the input
        batch_size, max_lang_num, seq_length = input_ids.shape
        input_ids = input_ids.view(batch_size*max_lang_num, seq_length)
        bert_attention_mask = bert_attention_mask.view(
            batch_size*max_lang_num, seq_length)

        # text encode (batch_size * max_lang_num, seq_length, hidden)
        lang_fea = self.encode(input_ids=input_ids,
                               attention_mask=bert_attention_mask)

        # whole text embedding
        lang_emb = lang_fea[:, 0, :]

        attention_mask = ~bert_attention_mask.bool().unsqueeze(1).unsqueeze(1).cuda()

        data_dict["lang_fea"] = lang_fea
        data_dict["lang_emb"] = lang_emb  # cls embedding
        data_dict['attention_mask'] = attention_mask

        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(lang_emb)
        return data_dict

    def mlm(self, data_dict):
        """
            MLM tasks
        """
        if self.use_distil:
            raise NotImplementedError(
                "MLM tasks not implemented for distilbert.")
        bbox_feature = data_dict['bbox_feature']
        input_ids = data_dict['input_ids']
        bert_attention_mask = data_dict['bert_attention_mask']
        positive_labels = data_dict['positive_labels']

        # point cloud embedding and attention masks
        batch_size, num_proposal, hidden_size = bbox_feature.shape
        bbox_feature = self.select(bbox_feature, batch_size, num_proposal,
                                hidden_size, positive_labels)

        # language embedding and attention masks
        batch_size, max_lang_num, seq_length = input_ids.shape
        input_ids = input_ids.view(batch_size*max_lang_num, seq_length)

        # attention masks
        bert_attention_mask = bert_attention_mask.view(
            batch_size*max_lang_num, seq_length)

        # random select batch_size samples
        sample_ids = torch.randint(
            batch_size*max_lang_num, (1, batch_size)).view(batch_size).cuda()
        sample_input_ids = input_ids[sample_ids]  # batch_size, seq_length
        sample_bert_attention_mask = bert_attention_mask[sample_ids]

        # construct sequence embedding within a scene
        sample_scenes = torch.floor_divide(sample_ids, max_lang_num).cuda()
        # batch_size, max_lang_num, hidden_size
        sample_bbox_feature = bbox_feature[sample_scenes]

        # mask input tokens
        sample_labels = sample_input_ids.clone()
        probability_matrix = torch.full(sample_labels.shape, self.mask_ratio)
        mlm_input_ids, sample_labels = self.mask(
            sample_input_ids, self.text_encoder.config.vocab_size, targets=sample_labels, probability_matrix=probability_matrix)

        # mlm text embedding
        lang_fea = self.encode(input_ids=mlm_input_ids,
                               attention_mask=sample_bert_attention_mask)
        mlm_feature = lang_fea

        # decoder
        for i in range(self.depth):
            # Get Attention Weight
            # if self.use_dist_weight_matrix:
            # center of the selected proposal
            #     objects_center = data_dict['pred_bbox_corner'].mean(dim=-2)
            #     objects_center = objects_center.view(
            #         batch_size*num_proposal, -1)
            #     print("object center", objects_center.shape)
            #     print("sample ids", sample_ids.shape)
            #     samples_center = objects_center[sample_ids]

            #     print("center", samples_center.shape)
            #     N_K = samples_center.shape[0]
            #     center_A = samples_center[None, :, :].repeat(N_K, 1, 1)
            #     center_B = samples_center[:, None, :].repeat(1, N_K, 1)
            #     print(center_A.shape)
            #     print(center_B.shape)
            #     center_dist = (center_A - center_B)
            #     dist = center_dist.pow(2)
            #     dist = torch.sqrt(torch.sum(dist, dim=-1))[None, :, :]
            #     print("dist", dist.shape)
            #     weights = torch.cat([center_dist, dist.permute(
            #         1, 2, 0)], dim=-1).detach()  # N N 4
            #     print("weight", weights.shape)
            #     dist_weights = self.dist_fc[i](
            #         weights).permute(2, 0, 1)
            #     attention_matrix_way = 'add'
            #     print("dist weight", dist_weights.shape)
            # else:
            dist_weights = None
            attention_matrix_way = 'mul'
            # print("mlm feature", mlm_feature.shape)
            # print("smaple_bbox_feature", sample_bbox_feature.shape)
            # print("dist_weight", dist_weights.shape)

            # construct postion embedding
            # pred_center = data_dict['pred_center']  # (B, num_proposal,3)
            # pred_box_size = data_dict['pred_size']  # (B. num_proposal,3)
            # pred_center = self.select(
            #     pred_center, batch_size, num_proposal, 3, positive_labels)  # (B, lan_num_max,3)
            # pred_box_size = self.select(
            #     pred_box_size, batch_size, num_proposal, 3, positive_labels)  # (B, lan_num_max,3)

            # # (B, lan_num_max,6)
            # pos_input = torch.cat([pred_center, pred_box_size], dim=-1)
            # # (B, lan_num_max, hidden_size)
            # pos_embed = self.pos_head[i](pos_input)
            # sample_pos_embed = pos_embed[sample_scenes]
            # sample_input = sample_bbox_feature + sample_pos_embed
            sample_input = sample_bbox_feature
            mlm_feature = self.cross_attn[i](
                mlm_feature, sample_input, sample_input, attention_weights=dist_weights,
                way=attention_matrix_way)
        pred = self.prediction(mlm_feature)

        mlm_loss = self.loss_fct(
            pred.view(-1, self.text_encoder.config.vocab_size), sample_labels.view(-1))

        data_dict['mlm_loss'] = mlm_loss
        return data_dict

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        """
        Masks the input tokens.
        """
        mlm_input_ids = input_ids.clone()
        masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[mlm_input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[mlm_input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            mlm_input_ids.shape, 0.8)).bool() & masked_indices
        mlm_input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            mlm_input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            vocab_size, mlm_input_ids.shape, dtype=torch.long).cuda()
        mlm_input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return mlm_input_ids, targets
        else:
            return mlm_input_ids

    def select(self, embeds, batch_size, num_proposal, hidden_size, positive_labels):
        """
            selects the embedding according to the positive labels and then reshape
            :param embeds: input embedding (batch_size, num_proposal, hidden_size)
            :param batch_size: batch_size
            :param num_proposal: proposal number
            :param hidden_size: hidden size
            :param positive_labels: positive pairs according to the text (1, lang_num_max)
        """
        out = embeds.view(batch_size*num_proposal, hidden_size)
        out = out[positive_labels]
        out = out.view(batch_size, -1, hidden_size)
        return out

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()