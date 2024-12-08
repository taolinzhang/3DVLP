import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lang_bert_module.bert.xbert import BertLMHeadModel, BertConfig, BertEmbeddings
from models.lang_bert_module.bert.tokenization_bert import BertTokenizer
from lib.configs.config_captioning import CONF
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer


class CaptionModule(nn.Module):
    def __init__(self, hidden_size=128, vocab_size=30522, depth=4):
        super().__init__()
        self.min_iou = CONF.TRAIN.MIN_IOU_THRESHOLD
        self.object_start_token = 104
        self.max_len = CONF.TRAIN.MAX_DES_LEN
        self.bert_config = BertConfig.from_json_file(CONF.CAPTION_CONFIG)
        self.text_decoder = BertLMHeadModel(self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(
            CONF.BERT_TEXT_ENCODER)
        self.embeddings = BertEmbeddings(self.bert_config)
        self.depth = 4
        self.caption_attn = nn.ModuleList(
            CrossAttentionDecoderLayer(hidden_size=hidden_size)for _ in range(self.depth))
        self.caption_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, data_dict, is_eval=False):
        if is_eval:
            data_dict = self.forward_eval(data_dict)
        else:
            data_dict = self.forward_train(data_dict)
        return data_dict

    def get_target_embeds(self, extended_object_feat, positive_labels, batch_size, lang_num_max, num_proposal):
        row_index = torch.mul(torch.ones([batch_size*lang_num_max, 1]), torch.arange(
            batch_size*lang_num_max).reshape(batch_size*lang_num_max, 1))
        row_index = row_index.view(
            batch_size*lang_num_max).long()  # (B*lang_num_max)
        col_index = positive_labels.view(
            batch_size*lang_num_max).long()  # (B*lang_num_max)
        # (B*lang_num_max, hidden)
        target_embeds = extended_object_feat[row_index, col_index]
        return target_embeds

    def get_mask(self, attention_mask, object_mask, batch_size, lang_num_max, num_proposal, seq_len):

        # seq_len, seq_len
        tril_mask = torch.tril(torch.ones(seq_len, seq_len)).type(
            torch.BoolTensor).cuda()

        # B*lang_num_max, seq_len, seq_len
        batch_tril_mask = tril_mask[None, :, :].repeat(
            batch_size*lang_num_max, 1, 1)

        # B*lang_num_max, seq_len, seq_len
        batch_attention_mask = attention_mask[:, None, :].repeat(1, seq_len, 1)

        # B*lang_num_max, seq_len, seq_len
        src_mask = batch_tril_mask * batch_attention_mask
        # B*lang_num_max, 1, seq_len, seq_len
        src_mask = src_mask.unsqueeze(1)

        # B*lang_num_max, seq_len, num_proposal
        src_trg_seq_mask = attention_mask[:,
                                          :, None].repeat(1, 1, num_proposal)
        src_trg_object_mask = object_mask[:, None, :].repeat(
            lang_num_max, seq_len, 1)
        src_trg_mask = src_trg_seq_mask * src_trg_object_mask
        # B*lang_num_max, 1, seq_len, num_proposal
        src_trg_mask = src_trg_mask.unsqueeze(1)
        # torch.set_printoptions(profile="full")
        # print("src_mask",src_mask[0][0])
        # print("src_trg_mask",src_trg_mask[0][0])

        return src_mask, src_trg_mask

    def get_extended_object_feat(self, object_feat, batch_size, lang_num_max, num_proposal):
        extended_object_feat = object_feat[:, None, :, :].repeat(1, lang_num_max, 1, 1).reshape(
            batch_size*lang_num_max, num_proposal, -1)
        return extended_object_feat

    def forward_train(self, data_dict):
        # (batch_size, num_proposal, hidden)
        object_feat = data_dict['bbox_feature']
        batch_size, num_proposal, hidden = object_feat.shape
        lang_num_max = data_dict["input_ids"].shape[1]

        # (B, lang_num_max, seq_len)
        attention_mask = data_dict['bert_attention_mask']
        input_ids = data_dict['input_ids']

        # (B, proposal)
        object_mask = data_dict['objectness_scores'].argmax(-1)

        # (B*lang_num_max, seq_len)
        input_ids = input_ids.view(batch_size*lang_num_max, -1)
        attention_mask = attention_mask.view(batch_size*lang_num_max, -1)

        # extend attention mask for object token
        extended_attention_mask = torch.cat(
            [torch.ones(batch_size*lang_num_max, 1).cuda(), attention_mask], dim=1)
        seq_len = extended_attention_mask.shape[1]

        # (batch_size*lang_num_max)
        positive_labels = data_dict["positive_labels"].detach()

        # (B*lang_num_max, num_proposal, hidden)
        extended_object_feat = self.get_extended_object_feat(
            object_feat, batch_size, lang_num_max, num_proposal)

        # # get src_mask and src_trg_mask
        # src_mask, src_trg_mask = self.get_mask(
        #     extended_attention_mask, object_mask, batch_size, lang_num_max, num_proposal, seq_len)

        # # (B*lang_num_max, max_len, hidden)
        # inputs_embeds = self.embeddings(input_ids)

        # # (B*lang_num_max, hidden)
        # target_embeds = self.get_target_embeds(
        #     extended_object_feat, positive_labels, batch_size, lang_num_max, num_proposal)
        # target_embeds = target_embeds.view(batch_size*lang_num_max, 1, -1)

        # # add target_embeds to inputs_embeds
        # inputs_embeds = torch.cat([target_embeds, inputs_embeds], dim=1)

        # feature = inputs_embeds
        # for i in range(self.depth):
        #     feature = self.caption_attn[i](
        #         feature, extended_object_feat, extended_object_feat, src_mask=src_mask, src_trg_mask=src_trg_mask)
        # data_dict["cap_feat"] = feature

        # pred_feature = feature[:, :-1, :]
        # lang_cap = self.caption_cls(pred_feature)
        # data_dict["lang_cap"] = lang_cap
        # ============================================

        inputs_embeds = self.get_inputs_embeds(input_ids)
        target_embeds = self.get_target_embeds(
            extended_object_feat, positive_labels, batch_size, lang_num_max, num_proposal)
        target_embeds = target_embeds.view(batch_size*lang_num_max, 1, -1)

        # set query embedding to the target_embeds
        inputs_embeds = torch.cat([target_embeds, inputs_embeds], dim=1)

        # extend attention mask
        object_attention_mask = torch.ones(batch_size*lang_num_max, 1).cuda()
        attention_mask = torch.cat(
            [object_attention_mask, attention_mask], dim=1)

        # extend labels
        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)
        # decoder_targets = decoder_targets.masked_fill(~good_bbox_masks, -100)

        object_token = -100 * \
            torch.ones(batch_size*lang_num_max, 1).long().cuda()

        decoder_targets = torch.cat([object_token, decoder_targets], dim=1)

        object_atts = torch.ones(
            extended_object_feat.size()[:-1], dtype=torch.long).cuda()

        decoder_output = self.text_decoder(input_ids=None,
                                           inputs_embeds=inputs_embeds,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=extended_object_feat,
                                           encoder_attention_mask=object_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        # ignore object token
        data_dict["lang_cap"] = decoder_output.logits[:, 1:-1, :]
        # captions = self.tokenizer.batch_decode(decoder_output.logits[:, 1:].argmax(-1), skip_special_tokens=True)
        # print("captions", captions)
        # ============================================
        return data_dict

    @torch.no_grad()
    def forward_eval(self, data_dict, num_beams=1):
        # (batch_size, num_proposal, hidden)
        object_feat = data_dict['bbox_feature']
        batch_size, num_proposal, hidden = object_feat.shape

        # # (B*num_proposal, 1, hidden)
        # object_feat = object_feat.view(batch_size*num_proposal, 1, -1)

        # # (B*num_proposal, 1)
        # cls_tokens = torch.ones(
        #     batch_size*num_proposal, 1).fill_(self.tokenizer.cls_token_id).long().cuda()
        # # (B*num_proposal, 1, hidden)
        # cls_embeds = self.embeddings(cls_tokens)

        # # (B*num_proposal, num_proposal, hidden)
        # extended_object_feat = self.get_extended_object_feat(
        #     object_feat, batch_size, num_proposal, num_proposal)

        # (B, proposal)
        object_mask = data_dict['objectness_scores'].argmax(-1)

        # # (B*num_proposal, 2, hidden)
        # inputs_embeds = torch.cat([object_feat, cls_embeds], dim=1)
        # outputs = cls_tokens
        # for i in range(self.max_len):
        #     seq_len = inputs_embeds.shape[1]
        #     feature = inputs_embeds
        #     # (B*num_proposal, 1, seq_len, seq_len)
        #     src_mask = torch.tril(torch.ones(seq_len, seq_len)).type(
        #         torch.BoolTensor).cuda().unsqueeze(0).repeat(batch_size*num_proposal, 1, 1).unsqueeze(1)
        #     # (B*num_proposal, 1, seq_len, num_proposal)
        #     src_trg_mask = object_mask[:, None, :].repeat(
        #         num_proposal, seq_len, 1).unsqueeze(1)
        #     for i in range(self.depth):
        #         feature = self.caption_attn[i](
        #             feature, extended_object_feat, extended_object_feat, src_mask=src_mask, src_trg_mask=src_trg_mask)
        #     # (B*num_proposal, hidden)
        #     pred_feature = feature[:, -1, :]
        #     # (B*num_proposal)
        #     logits = self.caption_cls(pred_feature)
        #     next_word = logits.argmax(-1)
        #     next_word = next_word.data.unsqueeze(1)
        #     outputs = torch.cat([outputs, next_word], dim=1)
        #     inputs_embeds = torch.cat(
        #         [inputs_embeds, self.embeddings(next_word)], dim=1)
        # outputs = outputs.view(batch_size, num_proposal, -1)
        # data_dict["lang_cap"] = outputs
        # =========================================
        input_ids = self.tokenizer.cls_token_id * \
            torch.ones(batch_size, 1).long().cuda()

        # object token
        object_token = torch.arange(
            self.object_start_token, self.object_start_token + batch_size).long().cuda().view(batch_size, 1)

        extended_input_ids = torch.cat([object_token, input_ids], dim=1)

        extended_object_feat = object_feat.repeat_interleave(
            num_beams, dim=0)

        model_kwargs = {"encoder_hidden_states": extended_object_feat,
                        "encoder_attention_mask": object_mask}

        batch_outputs = []
        for prop_id in range(num_proposal):
            # (B, hidden)
            target_embeds = object_feat[:, prop_id]
            # set query embedding to the target_embeds
            self.set_inputs_embeds(target_embeds)


            # print("input_ids", object_token.shape)
            # print("encoder_hidden_states", extended_object_feat.shape, "\n", "encoder_attention_mask",object_atts.shape)
            # beam search
            outputs = self.text_decoder.generate(input_ids=extended_input_ids,
                                                 max_length=CONF.TRAIN.MAX_DES_LEN+1,
                                                 min_length=CONF.TRAIN.MIN_DES_LEN+1,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.0,
                                                 **model_kwargs)
            outputs=outputs.unsqueeze(1)
            batch_outputs.append(outputs)
            # print("outputs", outputs.shape)
        batch_outputs = torch.cat(batch_outputs, dim=1)
        data_dict["lang_cap"] = batch_outputs[:, :, 1:]
        # ===============================
        return data_dict

    def get_inputs_embeds(self, input_ids):
        bert_embeds = self.text_decoder.get_input_embeddings()
        return bert_embeds(input_ids)

    @torch.no_grad()
    def set_inputs_embeds(self, embedding):
        self.text_decoder.bert.set_object_embeddings(
            value=embedding, start_token=self.object_start_token)
