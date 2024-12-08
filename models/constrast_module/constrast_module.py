import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_fn import *
from utils.box_util import box3d_diou_batch_tensor
from pytorch3d.ops.iou_box3d import box3d_overlap


def create_box_batch(center, size):
    extend_center = center[:, None, :].repeat(1, 8, 1)  # bs, 8, 3
    unit = torch.tensor([[[-1, -1, -1], [1, -1, -1], [1, 1, -1],
                          [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]]).cuda()
    extend_size = size[:, None, :]*unit/2  # bs, 8, 3
    box_batch = extend_center+extend_size
    return box_batch.float()


def SoftCrossEntropy(inputs, target):
    log_likelihood = -F.log_softmax(inputs, dim=-1)
    loss = torch.mean(torch.mul(log_likelihood, target))
    return loss


class NCELoss(nn.Module):
    def __init__(self, init_tau=0.07, clamp=4.6051):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor(
            [np.log(1.0 / init_tau)], dtype=torch.float32))
        self.clamp = clamp  # 4.6051 等价于CLAMP 100, 初始值是2.6593，

    def forward(self, logits, iou_matrix):
        # self.tau.data = torch.clamp(self.tau.data, 0, self.clamp)
        # logits = logits * self.tau.exp()
        loss_v = SoftCrossEntropy(logits, iou_matrix)
        loss_t = SoftCrossEntropy(logits.t(), iou_matrix)
        loss = (loss_v + loss_t) / 2
        return loss


class ContrastModule(nn.Module):
    def __init__(self, config, hidden=128):
        super().__init__()
        self.pc_proj = nn.Linear(hidden, hidden, bias=False)
        self.text_proj = nn.Linear(hidden, hidden, bias=False)
        self.nce_loss = NCELoss()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.pc_proj_iou = nn.Sequential(
            nn.Linear(hidden,hidden,bias=False)
        )

    def forward(self, data_dict):
        if data_dict["epoch"] < 50:
            data_dict["con_loss"] = torch.zeros(1)
            return data_dict
        pred_center = data_dict['pred_center'].detach()
        pred_box_size = data_dict['pred_size'].detach()
        features = data_dict["bbox_feature"]  # bs, num_proposal, hidden

        gt_center_list = data_dict['ref_center_label_list'].detach()  # (B,3)
        # B
        gt_heading_class_list = data_dict['ref_heading_class_label_list'].detach(
        )
        # B
        gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].detach(
        )
        # B
        gt_size_class_list = data_dict['ref_size_class_label_list'].detach()
        # B,3
        gt_size_residual_list = data_dict['ref_size_residual_label_list'].detach(
        )
        # convert gt bbox parameters to bbox corners
        batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
        batch_size, len_nun_max = gt_center_list.shape[:2]
        lang_num = data_dict["lang_num"]
        lang_emb = data_dict["lang_emb"]
        lang_num_max = lang_emb.shape[0]//batch_size
        lang_emb = lang_emb.view(batch_size, lang_num_max, -1)
        objectness_masks = data_dict['objectness_scores'].max(
            2)[1].float()

        lang_con_loss = 0.
        iou_con_loss = 0.
        for i in range(batch_size):
            pred_center_batch = pred_center[i]
            pred_box_size_batch = pred_box_size[i]
            gt_box_center, gt_box_size = self.config.param2obb_batch_tensor(
                gt_center_list[i][:, 0:3], gt_heading_class_list[i], gt_heading_residual_list[i], gt_size_class_list[i], gt_size_residual_list[i])
            object_index = torch.where(objectness_masks[i])[0]
            object_cnt=object_index.shape[0]
            features_batch = features[i][object_index]
            box_batch = create_box_batch(
                pred_center_batch[object_index], pred_box_size_batch[object_index])
            
            for j in range(len_nun_max):
                if j < lang_num[i]:
                    # convert the bbox parameters to bbox corners
                    lang_emb_batch = lang_emb[i][j][None,:]
                    gt_box_size_batch = gt_box_size[j][None, :]
                    gt_box_center_batch = gt_box_center[j][None, :]
                    gt_box_batch = create_box_batch(
                        gt_box_center_batch, gt_box_size_batch+1e-2)
                    try:
                        _, ious = box3d_overlap(
                            gt_box_batch, box_batch, eps=1e-7)  # 1, 256
                        ious = ious.view(-1)  # 256
                        target_mask_lang = (
                            ious > 0.25).float().unsqueeze(0).detach()
                        text_feat_norm_lang = F.normalize(
                            self.text_proj(lang_emb_batch), dim=-1)
                        box_feat_norm_lang = F.normalize(
                            self.pc_proj(features_batch), dim=-1)
                        sim_lang = torch.mm(
                            text_feat_norm_lang, box_feat_norm_lang.t())  # 1,256
                        lang_con_loss += self.nce_loss(sim_lang, target_mask_lang)

                        target_mask_iou = (ious > 0.25).float().detach()
                        target_mask_iou = target_mask_iou.unsqueeze(0).repeat(object_cnt,1)*target_mask_iou.unsqueeze(1).repeat(1,object_cnt).detach()
                        box_feat_norm_iou = F.normalize(self.pc_proj_iou(features_batch),dim=-1)
                        sim_iou = torch.mm(
                            box_feat_norm_iou, box_feat_norm_iou.t()) # object_cnt, object_cnt
                        iou_con_loss += self.nce_loss(sim_iou, target_mask_iou)
                    except Exception as e:
                        print("Error:", e)

        lang_con_loss /= batch_size
        iou_con_loss /= batch_size
        data_dict["lang_con_loss"] = lang_con_loss
        data_dict["iou_con_loss"] =iou_con_loss 
        return data_dict
    