import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from models.pointpillars.ops import Voxelization


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(
                pts)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        # (p1 + p2 + ... + pb, num_points, c)
        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(
            npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        # (p1 + p2 + ... + pb, 1 + 3)
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int(
            (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int(
            (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        # (p1 + p2 + ... + pb, num_points, 3)
        offset_pt_center = pillars[:, :, :3] - torch.sum(
            pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        # 2. calculate offset to the pillar center
        # (p1 + p2 + ... + pb, num_points, 1)
        x_offset_pi_center = pillars[:, :, :1] - \
            (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - \
            (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        # 3. encoder
        # (p1 + p2 + ... + pb, num_points, 9)
        features = torch.cat(
            [pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(
            device)  # (num_points, )
        # (num_points, p1 + p2 + ... + pb)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        # (p1 + p2 + ... + pb, num_points)
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        # 5. embedding
        # (p1 + p2 + ... + pb, 9, num_points)
        features = features.permute(0, 2, 1).contiguous()
        # (p1 + p2 + ... + pb, out_channels, num_points)
        features = F.relu(self.bn(self.conv(features)))
        # (p1 + p2 + ... + pb, out_channels)
        pooling_features = torch.max(features, dim=-1)[0]

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros(
                (self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        # (bs, in_channel, self.y_l, self.x_l)
        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[5, 2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(
                in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(
                out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(
                    nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(
                    out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    out_channels[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(
                out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 300, 240), (bs, 128, 150, 120), (bs, 256, 75, 60)]
        return: (bs, 384, 300, 240)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class PointPillars(nn.Module):
    def __init__(self,
                 #  voxel_size=[0.16, 0.16, 4],
                 #  point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 voxel_size=[0.05, 0.05, 10],
                 point_cloud_range=[-12, -15, -4, 12, 15, 6],
                 max_num_points=32,
                 max_voxels=(16000, 40000),
                 use_multiview=True):
        super().__init__()
        self.use_multiview = use_multiview
        self.in_channel = 140 if self.use_multiview else 9
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=self.in_channel,
                                            out_channel=64)
        self.backbone = Backbone(in_channel=64,
                                 out_channels=[64, 128, 128, 128],
                                 layer_nums=[3, 5, 5, 5])
        # self.neck = Neck(in_channels=[64, 128, 256],
        #                  upsample_strides=[1, 2, 4, 8],
        #                  out_channels=[128, 128, 128])
        # self.pooler = RoIAlign(1, spatial_scale=0.25, sampling_ratio=2)

    def forward(self, data_dict):
        batched_pts = data_dict["point_clouds"]
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(
            batched_pts)
        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(
            pillars, coors_batch, npoints_per_pillar)

        # (bs, c, w, h)
        # xs:  [(bs, 64, 300, 240), (bs, 128, 150, 120), (bs, 256, 75, 60)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 300, 240)
        # x = self.neck(xs)
        x= xs[-1]

        data_dict["pc_emb"] = x

        # lang_emb = data_dict["lang_emb"]  # (bs*lang_num_max, hidden)
        # pc_emb = xs[1].permute(0, 3, 2, 1).contiguous()
        # batch_size, h, w, hidden = pc_emb.shape  # (bs, h, w, hidden)
        # lang_num_max = lang_emb.shape[0]//batch_size

        # # reshape for batch matrix multiply
        # pc_emb = pc_emb.view(batch_size, h*w, hidden)  # (bs, h*w, hidden)
        # pc_emb = F.normalize(pc_emb, dim=-1)

        # lang_emb = F.normalize(lang_emb, dim=-1)  # (bs*lang_num_max, hidden)
        # # (bs, lang_num_max, hidden)
        # lang_emb = lang_emb.view(batch_size, lang_num_max, hidden)
        # # (bs, hidden, lang_num_max)
        # lang_emb = lang_emb.permute(0, 2, 1).contiguous()

        # # (bs, h*w, lang_num_max)
        # attn_score = torch.bmm(pc_emb, lang_emb)
        # # (bs, h, w, lang_num_max)
        # attn_score = attn_score.view(batch_size, h, w, lang_num_max)
        # attn_score = attn_score.permute(0, 3, 1, 2).contiguous()
        # data_dict["attn_score"] = attn_score  # (bs, lang_num_max, h, w)

        # attn_weight = self.get_attn_weight(data_dict)
        # # # (bs, num_proposal, lang_num_max)
        # data_dict["attn_weight"] = attn_weight

        return data_dict

    def get_attn_weight(self, data_dict):
        """
            cal attention weight for each proposal
        """
        attn_score = data_dict['attn_score']  # (bs, lang_num_max, h, w)
        pred_center = data_dict['pred_center']  # (B, num_proposal)
        pred_box_size = data_dict['pred_size']  # (B, num_proposal, 3)

        batch_size, num_proposal, _ = pred_center.shape
        batch_size, lang_num_max, h, w = attn_score.shape

        # (bs, num_proposal, 1)
        x_min = ((pred_center[:, :, 0]-pred_box_size[:, :, 0]/2 - self.point_cloud_range[0]
                  )/self.voxel_size[0]).view(batch_size, num_proposal, 1)
        y_min = ((pred_center[:, :, 1]-pred_box_size[:, :, 1]/2 - self.point_cloud_range[1]
                  )/self.voxel_size[1]).view(batch_size, num_proposal, 1)
        x_max = ((pred_center[:, :, 0]+pred_box_size[:, :, 0]/2 - self.point_cloud_range[0]
                  )/self.voxel_size[0]).view(batch_size, num_proposal, 1)
        y_max = ((pred_center[:, :, 1]+pred_box_size[:, :, 1]/2 - self.point_cloud_range[1]
                  )/self.voxel_size[1]).view(batch_size, num_proposal, 1)

        pred_box = torch.cat(
            [x_min, y_min, x_max, y_max], dim=-1)  # (bs, num_proposal, 4)
        pred_rois = self.bbox2roi(pred_box)  # (bs*num_proposal, 5)

        # (batch_size*num_proposal,lang_num_max, 1, 1)
        attn_weight = self.pooler(attn_score, pred_rois)
        # (batch_size, num_proposal, lang_num_max)
        attn_weight = attn_weight.view(
            batch_size, num_proposal, lang_num_max)
        # (batch_size, lang_num_max, num_proposal)
        attn_weight = attn_weight.permute(0, 2, 1).contiguous()
        # (batch_size*lang_num_max, num_proposal)
        attn_weight = attn_weight.view(
            batch_size*lang_num_max, num_proposal)

        return attn_weight

    def bbox2roi(self, bbox_list):
        """Convert a list of bboxes to roi format.
        Args:
            bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
                of images.
        Returns:
            Tensor: shape (bs*num_proposal, 5), [batch_ind, x1, y1, x2, y2]
        """
        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            if bboxes.size(0) > 0:
                img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
                rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
            else:
                rois = bboxes.new_zeros((0, 5))
            rois_list.append(rois)
        rois = torch.cat(rois_list, 0)
        return rois
