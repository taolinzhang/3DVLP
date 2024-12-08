import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer
from models.transformer.utils import PositionWiseFeedForward
from models.vqa.mcan_module import MCAN_ED, AttFlat, LayerNorm


class MatchModule(nn.Module):
    def __init__(self, hidden_size=128, mcan_num_layers=4,
                 mcan_num_heads=4,
                 mcan_pdrop=0.1,
                 mcan_flat_mlp_size=512,
                 mcan_flat_glimpses=1,
                 mcan_flat_out_size=128,):
        super().__init__()
        # Feature projection
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        self.object_feat_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        # Esitimate confidence
        self.object_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # Fusion backbone
        self.fusion_backbone = MCAN_ED(
            hidden_size, num_heads=mcan_num_heads, num_layers=mcan_num_layers, pdrop=mcan_pdrop)
        self.fusion_norm = LayerNorm(mcan_flat_out_size)

        # QA head
        self.attflat_visual = AttFlat(
            hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_lang = AttFlat(
            hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        # self.answer_cls = nn.Sequential(
        #     nn.Linear(mcan_flat_out_size, hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(answer_pdrop),
        #     nn.Linear(hidden_size, num_answers)
        # )

    def forward(self, data_dict):
        # sentences cnt within a batch
        lang_num_max = data_dict["input_ids"].shape[1]
        # word embeddings after LSTM (batch_size, num_words(max_question_length), hidden_size * num_dir)
        lang_feat = data_dict["lang_fea"]
        lang_mask = None

        # batch_size, num_proposal, proposal_size (128)
        object_feat = data_dict['bbox_feature']
        object_mask = None
        batch_size, num_proposal, hidden = object_feat.shape

        # --------- QA BACKBONE ---------
        # Pre-process Lanauge & Image Feature
        # batch_size*lang_num_max, num_words, hidden_size           torch.Size([16, 13, 128])
        # lang_feat = self.lang_feat_linear(lang_feat)
        # batch_size, num_proposal, hidden_size  torch.Size([2, 256, 128])
        # object_feat = self.object_feat_linear(object_feat)
        # batch_size*lang_num_max, num_proposal, hidden_size  torch.Size([16, 256, 128])
        object_feat = object_feat[:, None, :, :].repeat(1, lang_num_max, 1, 1).reshape(
            batch_size*lang_num_max, num_proposal, -1)

        # QA Backbone (Fusion network)
        lang_feat, object_feat = self.fusion_backbone(
            lang_feat,
            object_feat,
            lang_mask,
            object_mask,
        )
        # batch_size, num_proposal, 1
        object_score_mask = data_dict['objectness_scores'].max(
            2)[1].float().unsqueeze(2)
        # batch_size*lang_num_max, num_proposal, 1
        object_score_mask = object_score_mask[:, None, :, :].repeat(1, lang_num_max, 1, 1).reshape(
            batch_size*lang_num_max, num_proposal, -1)
        # object_conf_feat = object_feat

        # lang_flat_feat = self.attflat_lang(          # torch.Size([16, 512])
        #     lang_feat,
        #     lang_mask
        # )

        # object_flat_feat = self.attflat_visual(      # torch.Size([16, 512])
        #     object_feat,
        #     object_mask
        # )
        # # torch.Size([16, 128])
        # fuse_feat = self.fusion_norm(lang_flat_feat + object_flat_feat)
        # fuse_feat = fuse_feat[:, None, :].repeat(
        #     1, num_proposal, 1)  # torch.Size([16, 256, 128])
        # object_feat = object_feat + fuse_feat

        object_conf_feat = object_feat * object_score_mask
        conf = self.object_cls(object_conf_feat).squeeze(-1)
        data_dict["cluster_ref"] = conf
        return data_dict
