import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer
from models.transformer.utils import PositionWiseFeedForward
import random
import numpy as np

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=128, head=4, use_lang_emb=False, use_pc_encoder=False, use_match_con_loss=False, depth=2, use_reg_head=False):
        super().__init__()
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.use_lang_emb = use_lang_emb
        self.use_pc_encoder = use_pc_encoder
        self.depth = depth
        self.use_reg_head = use_reg_head

        # self.match = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, 1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.PReLU(),
        #     nn.Conv1d(hidden_size, hidden_size, 1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.PReLU(),
        #     nn.Conv1d(hidden_size, 1, 1)
        # )
        self.match = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(hidden_size, 1)
        )
        if self.use_reg_head:
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 6),
                nn.Sigmoid()
            )
        self.lang_emb_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, num_proposals, 1)
        )
        # self.conf_proj = nn.Sequential(
        #     nn.Linear(num_proposals, num_proposals)
        # )
        # self.grounding_cross_attn = MultiHeadAttention(
        #     d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v
        self.grounding_cross_attn = nn.ModuleList(
            CrossAttentionDecoderLayer(hidden_size=hidden_size)for _ in range(self.depth))
        self.lang_emb_cross_attn = MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v
        self.loss_fn = nn.CrossEntropyLoss()
        self.box_con_proj = nn.Linear(hidden_size, hidden_size)
        self.lang_con_proj = nn.Linear(hidden_size, hidden_size)
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.use_match_con_loss = use_match_con_loss


    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        objectness_masks = data_dict['objectness_scores'].max(
            2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        # batch_size, num_proposals, feat_size
        features = data_dict["bbox_feature"]

        batch_size, num_proposal = features.shape[:2]
        len_nun_max = data_dict["input_ids"].shape[1]
        # objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(
                2, 1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]],
                             :] = obj_features[j:j + total_len - obj_lens[i], :]

        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(
            batch_size*len_nun_max, num_proposal, -1)
        # if self.training:
        #     lang_fea = data_dict["mlm_lang_fea"]
        # else:
        lang_fea = data_dict["lang_fea"]
        lang_fea = lang_fea[:,1:]

        # cross-attention
        # feature1 = self.grounding_cross_attn(
        #     feature1, lang_fea, lang_fea)  # (B*lang_num_max, 256, hidden)
        for i in range(self.depth):
            feature1 = self.grounding_cross_attn[i](
                feature1, lang_fea, lang_fea)  # (B*lang_num_max, 256, hidden)
        data_dict["cross_box_feature"] = feature1

        # match
        feature1_agg = feature1
        feature1_agg = feature1_agg.view(
            batch_size*len_nun_max*num_proposal, -1)
        # feature1_agg = feature1_agg.permute(0, 2, 1).contiguous()
        confidence1 = self.match(feature1_agg).squeeze(1)
        confidence1 = confidence1.view(batch_size*len_nun_max, num_proposal)

        # match by lang_emb
        if self.use_lang_emb:
            lang_emb = data_dict["lang_emb"]
            lang_num_max = lang_emb.shape[0]//batch_size
            lang_emb = lang_emb.view(batch_size, lang_num_max, -1)
            lang_emb_feature = self.lang_emb_cross_attn(
                lang_emb, feature0, feature0)
            lang_emb_feature = lang_emb_feature.view(
                batch_size*lang_num_max, -1, 1).contiguous()
            confidence2 = self.lang_emb_proj(lang_emb_feature).squeeze(2)

        # (batch_size*lang_num_max, num_proposal)
        confidence = confidence1+confidence2 if self.use_lang_emb else confidence1

        data_dict["cluster_ref"] = confidence

        if self.use_reg_head:
            # restrict the value in [-0.05, 0.05]
            box_reg = self.reg_head(feature1_agg)*0.1-0.05
            box_reg = box_reg.view(batch_size, len_nun_max, num_proposal, 6)
            data_dict['pred_center_reg'] = box_reg[..., 0:3]
            data_dict['pred_size_reg'] = box_reg[..., 3:6]

        return data_dict
