"""
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

import lib.pointnet2.pointnet2_utils
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
# from utils.box_util import get_3d_box_batch_of_rois_tensor, rotz_batch_pytorch
from utils.box_util import get_3d_box_batch, rotz_batch_pytorch
from models.proposal_module.ROI_heads.roi_heads import StandardROIHeads


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, mask_box=False, use_kl_loss=False, use_vote_weight=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.mask_box = mask_box
        self.use_kl_loss = use_kl_loss
        self.use_vote_weight=use_vote_weight

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.proposal = StandardROIHeads(
            num_heading_bin=num_heading_bin, num_class=num_class, seed_feat_dim=256, use_kl_loss=self.use_kl_loss)
        
        if self.use_vote_weight:
            self.votes_weight_predictor = nn.Sequential(  # 128 128 128 1
                nn.Conv1d(256, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                # nn.Conv1d(128, 128, kernel_size=1),
                # nn.BatchNorm1d(128),
                # nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, kernel_size=1),
                nn.Sigmoid()
                )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.use_vote_weight:
            vote_weights = self.votes_weight_predictor(features)
            data_dict["vote_weights"]=vote_weights
            features = features*vote_weights

        # Farthest point sampling (FPS) on votes
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)

        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(
            0, 2, 1).contiguous()  # (batch_size, num_proposal, 128)
        # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        data_dict['aggregated_vote_inds'] = sample_inds

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(features, data_dict)
        data_dict = self.decode_scores(data_dict)
        # net = self.proposal(features)
        # data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox
        aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]  # (B,K,3)
        pred_heading_class = torch.argmax(
            data_dict["heading_scores"], -1)  # B,num_proposal
        pred_heading_residual = torch.gather(
            data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        rois = data_dict['rois'].contiguous()

        pred_heading = pred_heading_class.float() * (2.0*np.pi/self.num_heading_bin) + \
            pred_heading_residual[..., 0]
        data_dict['pred_heading'] = pred_heading

        pred_box_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, N, 3)

        bsize = pred_box_size.shape[0]
        num_proposal = pred_box_size.shape[1]

        # Compute pred center xyz
        vote_xyz = (rois[:, :, 0:3] - rois[:, :, 3:6]) / 2  # (B, N, 3)
        R = rotz_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        # R = roty_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        vote_xyz = torch.matmul(
            vote_xyz.reshape(-1, 3).unsqueeze(1), R).squeeze(2)  # (B, N, 3)
        vote_xyz = vote_xyz.view(bsize, num_proposal, 3)
        pred_center = aggregated_vote_xyz - vote_xyz

        if self.mask_box and self.training:
            pred_center, pred_box_size = self.mask(pred_center, pred_box_size)
        data_dict['pred_size'] = pred_box_size
        data_dict['pred_center'] = pred_center

        # batch_size, num_proposals, 8, 3
        pred_bboxes = get_3d_box_batch(pred_box_size.detach().cpu().numpy(),
                                       pred_heading.detach().cpu().numpy(), pred_center.detach().cpu().numpy())
        pred_bboxes = torch.from_numpy(pred_bboxes).float().to(
            pred_center.device)  # .reshape(bsize, num_proposal, 8, 3)
        data_dict['pred_bbox_corner'] = pred_bboxes

        # Testing Scripts
        # center = pred_center[0, 13].cpu().numpy()
        # heading = pred_heading[0, 13].cpu().numpy()
        # size = pred_box_size[0, 13].cpu().numpy()
        # from utils.box_util import get_3d_box
        # newbox = get_3d_box(size, heading, center)

        # print(newbox, pred_bboxes[0, 13])
        # import ipdb
        # ipdb.set_trace()
        # # print(pred_center, pred_size)
        return data_dict

    def decode_scores(self, data_dict):
        """
        decode the predicted parameters for the bounding boxes

        """
        # processed box info
        # bounding box corner coordinates
        data_dict = self.decode_pred_box(data_dict)
        data_dict["pred_bbox_feature"] = data_dict["aggregated_vote_features"]
        # Not Useful
        data_dict["pred_bbox_mask"] = data_dict["objectness_scores"].argmax(-1)
        data_dict["pred_bbox_sems"] = data_dict["sem_cls_scores"].argmax(-1)

        return data_dict

    def mask(self, pred_center, pred_box_size):
        batch_size, num_proposal, _ = pred_center.shape
        # mask 30% box
        mask_index = torch.bernoulli(torch.full(
            [batch_size, num_proposal], 0.3)).bool().cuda()
        batch_mask_index = mask_index[:, :, None].repeat(1, 1, 3)

        # mean=0, sigma=2 for center
        random_center = (torch.randn([batch_size, num_proposal, 3])/2).cuda()
        # mean=1, sigma=1 for size
        random_size = (1+torch.randn([batch_size, num_proposal, 3])).cuda()

        mask_center = (~batch_mask_index)*pred_center + \
            batch_mask_index*random_center
        mask_size = (~batch_mask_index)*pred_box_size + \
            batch_mask_index*random_size

        return mask_center, mask_size
