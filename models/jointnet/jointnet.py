# coding=utf8
import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.lang_bert_module.lang_bert_module import LangBertModule
from models.proposal_module.proposal_module_fcos import ProposalModule
from models.proposal_module.relation_module import RelationModule
from models.refnet.match_module import MatchModule
# from models.match_module.match_module import MatchModule
# from models.capnet.caption_module import SceneCaptionModule, TopDownSceneCaptionModule
from models.caption_module.caption_module import CaptionModule
from models.constrast_module.constrast_module import ContrastModule
from models.positive_match_module.positive_match_module import PositiveMatchModule
# from models.pointpillars.pointpillars import PointPillars
from models.answer_module.answer_module import AnswerModule
from models.caption_module.transformer_captioner import TransformerDecoderModel
from models.mlcvnet.backbone_module import Pointnet2Backbone as MLCVPointnet2Backbone
from models.mlcvnet.voting_module import VotingModule as MLCVVotingModule

class JointNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, vocabulary, embeddings=None,
                 input_feature_dim=0, num_proposal=128, num_locals=-1, vote_factor=1, sampling="vote_fps",
                 no_caption=False, use_topdown=False, query_mode="corner", num_graph_steps=0, use_relation=False,
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=768, ground_hidden_size=256, caption_hidden_size=512, dataset_config=None, use_con=False, use_distil=False, unfreeze=6, use_mlm=False,
                 use_lang_emb=False, mask_box=False, use_pc_encoder=False, use_match_con_loss=False, use_reg_head=False, use_kl_loss=False, use_answer=False,num_answers=0,use_mlcv_net=False,use_vote_weight=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.no_caption = no_caption
        self.dataset_config = dataset_config
        self.use_con = use_con
        self.use_distil = use_distil
        self.unfreeze = unfreeze
        self.use_mlm = use_mlm
        self.use_lang_emb = use_lang_emb
        self.mask_box = mask_box
        self.use_pc_encoder = use_pc_encoder
        self.use_match_con_loss = use_match_con_loss
        self.use_reg_head = use_reg_head
        self.use_kl_loss = use_kl_loss
        self.use_answer = use_answer
        self.use_mlcv_net = use_mlcv_net
        self.use_vote_weight = use_vote_weight

        # --------- PROPOSAL GENERATION ---------
        if self.use_mlcv_net:
            # Backbone point feature learning
            self.backbone_net = MLCVPointnet2Backbone(
                input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = MLCVVotingModule(self.vote_factor, 256)
        else:
            # Backbone point feature learning
            self.backbone_net = Pointnet2Backbone(
                input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(
            num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, mask_box=self.mask_box, use_kl_loss=self.use_kl_loss,use_vote_weight=self.use_vote_weight)

        self.relation = RelationModule(
            num_proposals=num_proposal, det_channel=128)  # bef 256
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangBertModule(
                num_class=num_class, use_lang_classifier=use_lang_classifier, pc_hidden_size=128, lang_hidden_size=128, unfreeze=self.unfreeze, use_distil=self.use_distil)

            self.positive_match = PositiveMatchModule()

            # --------- CONSTRAST LEARNING ---------
            if self.use_con:
                self.constrast = ContrastModule(config=self.dataset_config)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(
                1 + int(self.use_bidir)) * ground_hidden_size, det_channel=128, use_lang_emb=self.use_lang_emb, use_pc_encoder=self.use_pc_encoder, use_match_con_loss=self.use_match_con_loss, use_reg_head=self.use_reg_head)  # bef 256
            # self.match = MatchModule(hidden_size=128)

        if not no_caption:
            self.caption = TransformerDecoderModel(30522)
        
        if self.use_mlm:
            self.mlm = TransformerDecoderModel(30522)

        if self.use_answer:
            self.answer = AnswerModule(num_answers=num_answers)

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)
        data_dict = self.relation(data_dict)

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #           POINT CLOUD BRANCH        #
            #                                     #
            #######################################
            if self.use_pc_encoder:
                data_dict = self.pc_encoder(data_dict)

            #######################################
            #                                     #
            #           POSITIVE MATCHING         #
            #                                     #
            #######################################
            # data_dict = self.positive_match(
            #     data_dict, config=self.dataset_config)


            #######################################
            #                                     #
            #           CONSTRAST LEARNING        #
            #                                     #
            #######################################
            if self.use_mlm and not is_eval:
                data_dict = self.mlm.forward_mlm(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            # config for bbox_embedding
            data_dict = self.match(data_dict)
            #######################################
            #                                     #
            #           CONSTRAST LEARNING        #
            #                                     #
            #######################################
            if self.use_con:
                data_dict = self.constrast(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict,is_eval)
        
        if self.use_answer:
            data_dict = self.answer(data_dict)

        return data_dict
