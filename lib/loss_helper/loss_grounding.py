# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch_scatter

# sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from .loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch, box3d_diou_batch_tensor
from utils.box_util import rotz_batch_pytorch
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
# put larger weights on positive objectness
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]


@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, objectness_label, lang_num, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Note that true labels may be zero when iou rate < 0.25
    objectness_label: objects near the real bbox in the same scene (num_proposal)
    lang_num: valid proposal
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    len_nun_max, classes = true_labels.shape
    # lang_num
    lang_num_mask = torch.zeros(objectness_label.shape).cuda()
    lang_num_mask[:lang_num] = 1
    # set true labels to be true in obj_label
    obj_label = lang_num_mask*objectness_label
    true_mask = torch.sum(true_labels, dim=0).bool()
    obj_label = (~true_mask)*obj_label+true_mask
    divide = torch.sum(obj_label) - 1
    if divide == 0:
        divide = divide+1
    smooth_confidence = smoothing / divide
    batch_obj_label = obj_label[None, :].repeat(
        len_nun_max, 1).contiguous()
    assert (torch.min(obj_label - true_labels) >= 0)
    with torch.no_grad():
        smooth_label = torch.empty(true_labels.shape).cuda()
        smooth_label = torch.fill_(smooth_label, smooth_confidence)
        smooth_label = smooth_label * (batch_obj_label - true_labels) + \
            true_labels * confidence
    return smooth_label

def compute_vote_weight_loss(data_dict):
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1]  # B,num_seed,3
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds).float()
    vote_weights = data_dict["vote_weights"]
    vote_weights = vote_weights.view(batch_size,num_seed)
    loss_fn = nn.BCELoss()
    vote_weight_loss = loss_fn(vote_weights,seed_gt_votes_mask)
    return vote_weight_loss

def compute_attr_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """
    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1]  # B,num_seed,3
    vote_xyz = data_dict['vote_xyz']  # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    instance_labels = data_dict['instance_labels']

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(
        data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(
        batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes = seed_gt_votes[:, :, :3]  # 只取 1 个 gt vote
    seed_gt_votes += data_dict['seed_xyz']  # gt vote 差值加上 seed 的 xyz 坐标

    # Compute the min of min of distance
    # vote 出来的坐标 xyz bs, num_seed, 3
    # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_instance_labels = torch.gather(
        data_dict['instance_labels'], 1, seed_inds)
    attr_xyz = torch_scatter.scatter(
        vote_xyz, seed_instance_labels, dim=1, reduce="mean")
    row_index = torch.arange(batch_size).view(
        batch_size, 1).repeat(1, num_seed).cuda()
    col_index = seed_instance_labels
    attr_dist = torch.sum(
        torch.abs(vote_xyz-attr_xyz[row_index, col_index]),dim=-1)

    attr_loss = torch.sum(attr_dist * seed_gt_votes_mask.float()) / \
        (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return attr_loss


def compute_diou_loss(data_dict, config, no_reference=False, use_reg_head=False, use_kl_loss=False, debug=False):
    """ Compute cluster diou loss

    Args:
        data_dict: dict (read-only)

    Returns:
        diou_loss
    """
    # predicted bbox
    pred_heading = data_dict['pred_heading'].clone()
    pred_center = data_dict['pred_center'].clone()
    pred_box_size = data_dict['pred_size'].clone()

    if use_reg_head:
        # (B, lang_num_max, num_proposal, 3)
        pred_center_reg = data_dict['pred_center_reg'].clone()
        # (B, lang_num_max, num_proposal, 3)
        pred_box_size_reg = data_dict['pred_size_reg'].clone()

    if use_kl_loss:
        alpha = data_dict["alpha"].clone()  # (batch_size, num_proposal, 6)
        alpha_center = alpha[:, :, 0:3]
        alpha_size = alpha[:, :, 4:6]
        smooth_l1_criterion = nn.SmoothL1Loss()

    gt_center_list = data_dict['ref_center_label_list'].detach()  # (B,3)
    # B
    gt_heading_class_list = data_dict['ref_heading_class_label_list'].detach()
    # B
    gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].detach(
    )
    # B
    gt_size_class_list = data_dict['ref_size_class_label_list'].detach()
    # B,3
    gt_size_residual_list = data_dict['ref_size_residual_label_list'].detach()
    # convert gt bbox parameters to bbox corners
    batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
    batch_size, len_nun_max = gt_center_list.shape[:2]
    lang_num = data_dict["lang_num"]
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    # =========================================
    # for debug
    if debug:
        top_iou_rate = 0
        top_ind = []
        object_cat_list = data_dict["object_cat_list"].long()
        class_iou_rate = torch.zeros(18).cuda()
        class_cnt = torch.zeros(18).cuda()
        class_size = torch.zeros(18).cuda()
        pred_iou_rate_25_list = []
        pred_iou_rate_5_list = []
    # =========================================

    softmax_fn = nn.Softmax(dim=-1)
    criterion = SoftmaxRankingLoss()

    cluster_preds = data_dict["cluster_ref"].reshape(
        batch_size, len_nun_max, num_proposals)

    diou_loss = 0.
    loss = 0.
    kl_loss = 0.
    gt_labels = torch.zeros((batch_size, len_nun_max, num_proposals))
    for i in range(batch_size):
        objectness_masks = data_dict['objectness_scores'].max(
            2)[1].float()  # batch_size, num_proposals
        gt_box_center, gt_box_size = config.param2obb_batch_tensor(
            gt_center_list[i][:, 0:3], gt_heading_class_list[i], gt_heading_residual_list[i], gt_size_class_list[i], gt_size_residual_list[i])
        labels = torch.zeros((len_nun_max, num_proposals)).cuda()
        smooth_labels = torch.zeros((len_nun_max, num_proposals)).cuda() 
        sort_ind_list = []
        dious_labels = []
        pred_center_batch = pred_center[i]
        pred_box_size_batch = pred_box_size[i]
        if use_reg_head:
            # (lang_num_max, num_proposal, 3)
            pred_center_reg_batch = pred_center_reg[i]
            # (lang_num_max, num_proposal, 3)
            pred_box_size_reg_batch = pred_box_size_reg[i]
        if use_kl_loss:
            alpha_center_batch = alpha_center[i]
            alpha_size_batch = alpha_size[i]
            kl_ind = []
            kl_gt_center = []
            kl_gt_size = []
        for j in range(len_nun_max):
            if j < lang_num[i]:
                # convert the bbox parameters to bbox corners
                gt_box_size_batch = gt_box_size[j][None, :].repeat(
                    num_proposals, 1)
                gt_box_center_batch = gt_box_center[j][None, :].repeat(
                    num_proposals, 1)

                if use_reg_head:
                    ious, dious = box3d_diou_batch_tensor(
                        pred_center_batch+pred_center_reg_batch[j], pred_box_size_batch+pred_box_size_reg_batch[j], gt_box_center_batch, gt_box_size_batch)
                else:
                    ious, dious = box3d_diou_batch_tensor(
                        pred_center_batch, pred_box_size_batch, gt_box_center_batch, gt_box_size_batch)
                ious_np = ious.detach()
                dious_labels.append(dious)
                # ious = torch.pow(ious, 0.5)
                # # ious_weight = softmax_fn(ious)
                # ious_weight = ious
                # ious_weight = ious/torch.sum(ious, dim=-1)
                # dious = ious_weight * dious

                if data_dict["istrain"][0] == 1 and not no_reference and data_dict["random"] < 0.5:
                    ious = ious * objectness_masks[i]

                ious_ind = ious_np.argmax()
                max_ious = ious_np[ious_ind]
                sort_ious = torch.sort(ious_np)[0]
                if max_ious >= 0.25:
                    # treat the bbox with highest iou score as the gt
                    labels[j, ious_ind] = 1
                    if data_dict["epoch"] < 50:
                        smooth_mask = (ious>=0.25)
                        cnt = torch.sum(smooth_mask)
                        if cnt>=2 :
                            smooth_labels[j,smooth_mask]=0.05/(cnt-1)
                            smooth_labels[j, ious.argmax()] = 0.95
                        else:
                            smooth_labels[j, ious.argmax()] = 1
                    else:
                        smooth_labels[j, ious.argmax()] = 1
                    # treat iou rate as the target
                    max_iou_rate_25 += 1
                if max_ious >= 0.5:
                    max_iou_rate_5 += 1
                # ============================================
                # for debug
                if debug:
                    top_iou_rate+=sort_ious[-5:]
                    sort_ind = torch.argsort(ious_np)
                    sort_ind_list.append(sort_ind) # 5
                    mask_25 = torch.zeros(ious_np.shape).bool()
                    mask_25[ious_np >= 0.25] = True
                    pred_iou_rate_25_list.append(
                        torch.sum(mask_25)/mask_25.shape[0])
                    mask_5 = torch.zeros(ious_np.shape).bool()
                    mask_5[ious_np >= 0.5] = True
                    pred_iou_rate_5_list.append(torch.sum(mask_5)/mask_5.shape[0])
                    class_iou_rate[object_cat_list[i][j]] += ious_np[cluster_preds[i,j].argmax().long()]
                    class_size[object_cat_list[i][j]] += torch.prod(gt_box_size[j])
                    class_cnt[object_cat_list[i][j]] += 1
                # ============================================

                if use_kl_loss:
                    # (proposals, 6)
                    kl_ind.append(ious_ind)
                    kl_gt_center.append(gt_box_center[j])
                    kl_gt_size.append(gt_box_size[j])

        # cluster_labels = labels.detach()  # B proposals
        cluster_labels = smooth_labels.detach()
        gt_labels[i] = labels
        cluster_dious_labels = torch.stack(dious_labels).float()
        # reference loss
        loss += criterion(cluster_preds[i, :lang_num[i]],
                          cluster_labels[:lang_num[i]].float().clone())
        diou_loss += (torch.sum((1-cluster_dious_labels[:lang_num[i]])
                                * cluster_labels[:lang_num[i]].float().clone()))
        # print("diou_loss:",(1-cluster_dious_labels[:lang_num[i]])
        #                         * cluster_labels[:lang_num[i]].float().clone(),flush=True)
        # _, diou_index = torch.topk(
        #     cluster_preds[i], k=4, sorted=True) # (B*lang_num_max, 8)
        # =============================
        # for debug
        if debug:
            sort_ind_list = torch.stack(sort_ind_list).long()
            label_ind = cluster_preds[i, :lang_num[i]].argmax(dim=-1)
            label_ind = label_ind.reshape(-1,1)
            top_ind.append(torch.mean(torch.where(sort_ind_list[:lang_num[i]]==label_ind)[1].float()))
        # =============================
        

        if use_kl_loss:
            kl_index = torch.tensor(kl_ind).cuda()
            kl_gt_center = torch.stack(kl_gt_center).detach()
            kl_gt_size = torch.stack(kl_gt_size).detach()
            kl_pred_center = pred_center_batch[kl_index].detach()
            kl_pred_size = pred_box_size_batch[kl_index].detach()
            alpha_center_exp_batch = torch.exp(-alpha_center_batch)
            alpha_size_exp_batch = torch.exp(-alpha_size_batch)
            kl_loss_center = torch.sum(alpha_center_exp_batch*smooth_l1_criterion(
                kl_pred_center, kl_gt_center)+0.5*alpha_center_batch)
            kl_loss_size = torch.sum(
                alpha_size_exp_batch*smooth_l1_criterion(kl_pred_size, kl_pred_size)+0.5*alpha_size_batch)
            kl_loss += kl_loss_center+kl_loss_size

    data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / \
        torch.sum(lang_num).cpu().numpy()
    data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / \
        torch.sum(lang_num).cpu().numpy()
    # =============================
    # for debug
    if debug:
        class_cnt[class_cnt==0]+=1
        class_iou_rate /= class_cnt
        class_size /= class_cnt
        for i in range(0,18):
            data_dict[f'class_iou_rate_{i}'] = class_iou_rate[i].cpu().numpy() 
            data_dict[f'class_size_{i}'] = class_size[i].cpu().numpy()
        top_ind = torch.stack(top_ind).float()
        top_ind = torch.mean(top_ind)+1
        data_dict['top_ind'] = top_ind.cpu().numpy()
        data_dict['pred_iou_rate_0.25'] = torch.mean(
            torch.tensor(pred_iou_rate_25_list)).cpu().numpy()
        data_dict['pred_iou_rate_0.5'] = torch.mean(
            torch.tensor(pred_iou_rate_5_list)).cpu().numpy()
        for i in range(1,6):
            data_dict[f'top_iou_rate_{i}'] = top_iou_rate[5-i].cpu().numpy() / \
            torch.sum(lang_num).cpu().numpy()
    else:
        data_dict['top_ind'] = 0
        data_dict['pred_iou_rate_0.25'] = 0
        data_dict['pred_iou_rate_0.5'] = 0
        for i in range(0,18):
            data_dict[f'class_iou_rate_{i}'] = 0
            data_dict[f'class_size_{i}'] = 0
        for i in range(1,6):
            data_dict[f'top_iou_rate_{i}'] = 0
    # =============================
    

    cluster_labels = gt_labels.float().detach().cuda()  # B len_nun_max proposals
    loss = loss / batch_size
    diou_loss = diou_loss / batch_size
    data_dict['diou_loss'] = diou_loss
    if use_kl_loss:
        kl_loss = kl_loss/batch_size
        data_dict['kl_loss'] = kl_loss
    return data_dict, loss, cluster_preds, cluster_labels


def compute_reference_loss(data_dict, config, no_reference=False):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    # cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_heading = data_dict['pred_heading'].detach(
    ).cpu().numpy()  # B,num_proposal
    pred_center = data_dict['pred_center'].detach(
    ).cpu().numpy()  # (B, num_proposal)
    pred_box_size = data_dict['pred_size'].detach(
    ).cpu().numpy()  # (B, num_proposal, 3)

    gt_center_list = data_dict['ref_center_label_list'].cpu().numpy()  # (B,3)
    # B
    gt_heading_class_list = data_dict['ref_heading_class_label_list'].cpu(
    ).numpy()
    gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].cpu(
    ).numpy()  # B
    # B
    gt_size_class_list = data_dict['ref_size_class_label_list'].cpu().numpy()
    # B,3
    gt_size_residual_list = data_dict['ref_size_residual_label_list'].cpu(
    ).numpy()
    # convert gt bbox parameters to bbox corners
    batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
    batch_size, len_nun_max = gt_center_list.shape[:2]
    lang_num = data_dict["lang_num"]
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    if not no_reference:
        cluster_preds = data_dict["cluster_ref"].reshape(
            batch_size, len_nun_max, num_proposals)
    else:
        cluster_preds = torch.zeros(
            batch_size, len_nun_max, num_proposals).cuda()

    # print("cluster_preds",cluster_preds.shape)
    criterion = SoftmaxRankingLoss()
    loss = 0.
    gt_labels = np.zeros((batch_size, len_nun_max, num_proposals))
    for i in range(batch_size):
        objectness_masks = data_dict['objectness_scores'].max(
            2)[1].float().cpu().numpy()  # batch_size, num_proposals
        gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                              gt_heading_residual_list[i],
                                              gt_size_class_list[i], gt_size_residual_list[i])
        gt_bbox_batch = get_3d_box_batch(
            gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        labels = np.zeros((len_nun_max, num_proposals))
        iou_labels = np.zeros((len_nun_max, num_proposals))
        for j in range(len_nun_max):
            if j < lang_num[i]:
                # convert the bbox parameters to bbox corners
                pred_center_batch = pred_center[i]
                pred_heading_batch = pred_heading[i]
                pred_box_size_batch = pred_box_size[i]
                pred_bbox_batch = get_3d_box_batch(
                    pred_box_size_batch, pred_heading_batch, pred_center_batch)
                ious = box3d_iou_batch(pred_bbox_batch, np.tile(
                    gt_bbox_batch[j], (num_proposals, 1, 1)))

                if data_dict["istrain"][0] == 1 and not no_reference and data_dict["random"] < 0.5:
                    ious = ious * objectness_masks[i]

                ious_ind = ious.argmax()
                max_ious = ious[ious_ind]
                if max_ious >= 0.25:
                    # treat the bbox with highest iou score as the gt
                    labels[j, ious.argmax()] = 1
                    # treat iou rate as the target
                    iou_labels[j] = ious
                    max_iou_rate_25 += 1
                if max_ious >= 0.5:
                    max_iou_rate_5 += 1

        cluster_labels = torch.FloatTensor(labels).cuda()  # B proposals
        cluster_iou_labels = torch.FloatTensor(iou_labels).cuda()
        gt_labels[i] = labels
        # reference loss
        loss += criterion(cluster_preds[i, :lang_num[i]],
                          cluster_labels[:lang_num[i]].float().clone())
        # loss += criterion(cluster_preds[i, :lang_num[i]],
        #                   cluster_iou_labels[:lang_num[i]].float().clone())

    data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / \
        sum(lang_num.cpu().numpy())
    data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / \
        sum(lang_num.cpu().numpy())

    # print("max_iou_rate", data_dict['max_iou_rate_0.25'], data_dict['max_iou_rate_0.5'])
    cluster_labels = torch.FloatTensor(
        gt_labels).cuda()  # B len_nun_max proposals
    # print("cluster_labels", cluster_labels.shape)
    loss = loss / batch_size
    # print("ref_loss", loss)
    return data_dict, loss, cluster_preds, cluster_labels


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    object_cat_list = data_dict["object_cat_list"]
    batch_size, len_nun_max = object_cat_list.shape[:2]
    lang_num = data_dict["lang_num"]
    lang_scores = data_dict["lang_scores"].reshape(batch_size, len_nun_max, -1)
    loss = 0.
    for i in range(batch_size):
        num = lang_num[i]
        loss += criterion(lang_scores[i, :num], object_cat_list[i, :num])
    loss = loss / batch_size
    return loss


def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(
        data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(
        objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(
        objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(
        data_dict, config)
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    box_loss = box_loss + 20 * size_distance_loss

    # objectness; Nothing
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == data_dict["objectness_label"].long()).float(
    )*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    data_dict["obj_acc"] = obj_acc

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_distance_loss"] = size_distance_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        device = vote_loss.device()
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if reference:
        # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(
            data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # raise NotImplementedError('Only detection; not implemented')
        # # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(
            data_dict, config, no_reference=True)
        lang_count = data_dict['ref_center_label_list'].shape[1]
        # data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda().repeat(lang_count, 1)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["cluster_ref"] = objectness_label.new_zeros(
            objectness_label.shape).float().cuda().repeat(lang_count, 1)
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
        # data_dict['max_iou_rate_0.25'] = 0
        # data_dict['max_iou_rate_0.5'] = 0

    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    # loss = data_dict['vote_loss'] + 0.1 * data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1 * data_dict['sem_cls_loss'] + 0.03 * data_dict["ref_loss"] + 0.03 * data_dict["lang_loss"]
    loss = 0

    # Final loss function
    if detection:
        # sem_cls loss is included in the box_loss
        # detection_loss = detection_loss + 0.1 * data_dict['sem_cls_loss']
        detection_loss = data_dict["vote_loss"] + 0.1 * \
            data_dict["objectness_loss"] + 1.0*data_dict["box_loss"]
        detection_loss *= 10  # amplify
        loss = loss + detection_loss
    if reference:
        loss = loss + 0.3 * data_dict["ref_loss"]
    if use_lang_classifier:
        loss = loss + 0.3 * data_dict["lang_loss"]
    data_dict["loss"] = loss

    return data_dict
