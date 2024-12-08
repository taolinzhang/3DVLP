# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from lib.configs.config_captioning import CONF
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss
from utils.box_util import rotz_batch_pytorch

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_cap_loss(data_dict,config,weight,pad_token_id,tokenizer):

    pred_caps = data_dict["lang_cap"]
    num_words = pred_caps.size(1)

    target_caps = data_dict["input_ids"]
    batch_size,lang_num_max,_ = target_caps.shape
    target_caps=target_caps.view(batch_size*lang_num_max,-1)
    target_caps=target_caps[:, 1:num_words + 1]

    _, _, num_vocabs = pred_caps.shape

    assert pred_caps.shape[0:2] == target_caps.shape[0:2], print('ERROR!!! pred {} and tgt {} shape mismatch!'.format(
        pred_caps.shape[0:2], target_caps.shape[0:2]
    ))
    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # mask out bad boxes
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words)

    good_bbox_masks = good_bbox_masks.reshape(-1)
    cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)
    num_good_bbox = data_dict["good_bbox_masks"].sum()
    decode_pred = pred_caps.argmax(-1)
    # print("pred",decode_pred[0])
    # print("pred_caps ",tokenizer.batch_decode(decode_pred[0]))
    # print("target",target_caps[0])
    # print("target_caps ",tokenizer.batch_decode(target_caps[0]))
    # print()

    if num_good_bbox > 0:  # only apply loss on the good boxes
        pred_caps = pred_caps[data_dict["good_bbox_masks"]]  # num_good_bbox
        target_caps = target_caps[data_dict["good_bbox_masks"]]  # num_good_bbox

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)

        target_caps = target_caps.reshape(-1)  # num_good_bbox * (num_words - 1)
        masks = target_caps != pad_token_id
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()

    else:  # zero placeholder if there is no good box
        cap_acc = torch.zeros(1)[0].cuda()

    return cap_loss, cap_acc


def get_object_cap_loss(data_dict, config, weights, classify=True, caption=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    if classify:
        cls_loss, cls_acc = compute_object_cls_loss(data_dict, weights)

        data_dict["cls_loss"] = cls_loss
        data_dict["cls_acc"] = cls_acc
    else:
        data_dict["cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cls_acc"] = torch.zeros(1)[0].cuda()

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["cls_loss"] + data_dict["cap_loss"]

    # loss *= 10 # amplify

    data_dict["loss"] = loss

    return data_dict


def get_scene_cap_loss(data_dict, device, config, weights, 
    detection=True, caption=True, orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS):
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
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    box_loss = box_loss + 20 * size_distance_loss

    # objectness; Nothing
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
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
        device = vote_loss.device
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] =  torch.zeros(1)[0].to(device)

    if orientation:
        raise NotImplementedError()
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance:
        raise NotImplementedError()
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    # Final loss function
    # loss = data_dict["vote_loss"] + 0.1 * data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"] + data_dict["cap_loss"]

    if detection:
        loss = data_dict["vote_loss"] + 0.1*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"]
        # loss = data_dict["vote_loss"] + 1.0*data_dict["objectness_loss"] + 1.0*data_dict["box_loss"]
        loss *= 10 # amplify
        if caption:
            loss += 0.2*data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]
            # loss += data_dict["dist_loss"]
    else:
        loss = 0.2*data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]

    data_dict["loss"] = loss

    return data_dict

def get_object_cap_loss(data_dict, config, weights, classify=True, caption=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    if classify:
        cls_loss, cls_acc = compute_object_cls_loss(data_dict, weights)

        data_dict["cls_loss"] = cls_loss
        data_dict["cls_acc"] = cls_acc
    else:
        data_dict["cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cls_acc"] = torch.zeros(1)[0].cuda()

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["cls_loss"] + data_dict["cap_loss"]
    data_dict["loss"] = loss

    return data_dict
