import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
import random
from utils.utils_fn import concat_all_gather
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch, box3d_diou_batch_tensor


class PositiveMatchModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict, config, no_reference=False):
        data_dict["random"] = random.random()

        # predicted bbox
        pred_heading = data_dict['pred_heading'].clone()
        pred_center = data_dict['pred_center'].clone()
        pred_box_size = data_dict['pred_size'].clone()

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
        target_ious = []
        max_iou_rate_25 = 0
        max_iou_rate_5 = 0
        good_bbox_masks = []
        positive_labels=[]
        for i in range(batch_size):
            objectness_masks = data_dict['objectness_scores'].max(
                2)[1].float()  # batch_size, num_proposals
            gt_box_center, gt_box_size = config.param2obb_batch_tensor(
                gt_center_list[i][:, 0:3], gt_heading_class_list[i], gt_heading_residual_list[i], gt_size_class_list[i], gt_size_residual_list[i])
            pred_center_batch = pred_center[i]
            pred_box_size_batch = pred_box_size[i]
            for j in range(len_nun_max):
                # convert the bbox parameters to bbox corners
                gt_box_size_batch = gt_box_size[j][None, :].repeat(
                    num_proposals, 1)
                gt_box_center_batch = gt_box_center[j][None, :].repeat(
                    num_proposals, 1)

                ious, dious = box3d_diou_batch_tensor(
                    pred_center_batch, pred_box_size_batch, gt_box_center_batch, gt_box_size_batch)
                ious_np = ious.detach()
                ious = ious * objectness_masks[i]

                ious_ind = ious_np.argmax()
                max_ious = ious_np[ious_ind]
                good_bbox = False
                if max_ious >= 0.25:
                    # treat iou rate as the target
                    max_iou_rate_25 += 1
                    good_bbox = True
                if max_ious >= 0.5:
                    max_iou_rate_5 += 1
                target_ious.append(max_ious)
                good_bbox_masks.append(good_bbox)
                positive_labels.append(ious_ind)

        target_ious = torch.FloatTensor(target_ious).cuda()
        good_bbox_masks =torch.BoolTensor(good_bbox_masks).cuda() 
        positive_labels =torch.LongTensor(positive_labels).cuda()  
        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean(
        ) if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        data_dict["target_ious"] = target_ious
        data_dict["good_bbox_masks"] = good_bbox_masks
        data_dict["pred_ious"] = mean_target_ious
        data_dict["positive_labels"] = positive_labels
        return data_dict
