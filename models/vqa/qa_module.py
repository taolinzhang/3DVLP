import torch
import torch.nn as nn
from models.vqa.mcan_module import MCAN_ED, AttFlat, LayerNorm
from models.vqa.backbone import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.vqa.proposal import ProposalModule
from models.vqa.lang_module import LangModule

class ScanQA(nn.Module):
    def __init__(self, num_answers, 
        # proposal
        num_object_class, input_feature_dim,
        num_heading_bin, num_size_cluster, mean_size_arr, 
        num_proposal=256, vote_factor=1, sampling="vote_fps", seed_feat_dim=256, proposal_size=128,
        pointnet_width=1,
        pointnet_depth=2,        
        vote_radius=0.3, 
        vote_nsample=16,
        # qa
        #answer_cls_loss="ce",
        answer_pdrop=0.3,
        mcan_num_layers=2,
        mcan_num_heads=8,
        mcan_pdrop=0.1,
        mcan_flat_mlp_size=512, 
        mcan_flat_glimpses=1,
        mcan_flat_out_size=1024,
        # lang
        lang_use_bidir=False,
        lang_num_layers=1,
        lang_emb_size=300,
        lang_pdrop=0.1,
        bert_model_name=None,
        freeze_bert=False,
        finetune_bert_last_layer=False,
        # common
        hidden_size=128,
        # option
        use_object_mask=False,
        use_lang_cls=False,
        use_reference=False,
        use_answer=False,
    ):
        super().__init__() 

        # Option
        self.use_object_mask = use_object_mask
        self.use_lang_cls = use_lang_cls
        self.use_reference = use_reference
        self.use_answer = use_answer

        lang_size = hidden_size * (1 + lang_use_bidir)
        # Language encoding 
        self.lang_net = LangModule(num_object_class, use_lang_classifier=False, 
                                    use_bidir=lang_use_bidir, num_layers=lang_num_layers,
                                    emb_size=lang_emb_size, hidden_size=hidden_size, pdrop=lang_pdrop, 
                                    bert_model_name=bert_model_name, freeze_bert=freeze_bert,
                                    finetune_bert_last_layer=finetune_bert_last_layer)           

        # Ojbect detection
        self.detection_backbone = Pointnet2Backbone(input_feature_dim=input_feature_dim, 
                                                width=pointnet_width, depth=pointnet_depth,
                                                seed_feat_dim=seed_feat_dim)
        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal_net = ProposalModule(num_object_class, num_heading_bin, num_size_cluster, mean_size_arr, 
                                        num_proposal, sampling, seed_feat_dim=seed_feat_dim, proposal_size=proposal_size,
                                        radius=vote_radius, nsample=vote_nsample)   

        # Feature projection
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )
        self.object_feat_linear = nn.Sequential(
            nn.Linear(proposal_size, hidden_size),
            nn.GELU()
        )

        # Fusion backbone
        self.fusion_backbone = MCAN_ED(hidden_size, num_heads=mcan_num_heads, num_layers=mcan_num_layers, pdrop=mcan_pdrop)
        self.fusion_norm = LayerNorm(mcan_flat_out_size)

        # Esitimate confidence
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
        )

        # Language classifier
        self.lang_cls = nn.Sequential(
                nn.Linear(mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_object_class)
        )

        # QA head
        self.attflat_visual = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_lang = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.answer_cls = nn.Sequential(
                nn.Linear(mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(answer_pdrop),
                nn.Linear(hidden_size, num_answers)
        )

    def forward(self, data_dict):
        
        """
        data_dict:  torch.nonzero(data_dict["lang_out"].shape)

        dict_keys(['lang_feat',                     # torch.Size([16, 36, 300])
                   'point_clouds',                  # torch.Size([16, 40000, 7])
                   'lang_len',                      # torch.Size([16])
                   'center_label',                  # torch.Size([16, 128, 3])
                   'heading_class_label',           # torch.Size([16, 128])
                   'heading_residual_label',        # torch.Size([16, 128])
                   'size_class_label',              # torch.Size([16, 128])
                   'size_residual_label',           # torch.Size([16, 128, 3])
                   'num_bbox',                      # torch.Size([16])
                   'sem_cls_label',                 # torch.Size([16, 128])
                   'box_label_mask',                # torch.Size([16, 128])
                   'vote_label',                    # torch.Size([16, 40000, 9])
                   'vote_label_mask',               # torch.Size([16, 40000])
                   'scan_idx',                      # torch.Size([16])
                   'pcl_color',                     # torch.Size([16, 40000, 3])
                   'ref_box_label',                 # torch.Size([16, 128])
                   'ref_center_label',              # torch.Size([16, 3])
                   'ref_heading_class_label',       # torch.Size([16])
                   'ref_heading_residual_label',    # torch.Size([16])
                   'ref_size_class_label',          # torch.Size([16])
                   'ref_size_residual_label',       # torch.Size([16, 3])
                   'object_cat',                    # torch.Size([16])
                   'scene_id',                      # torch.Size([16])
                   'question_id',                   # torch.Size([16])
                   'load_time',                     # torch.Size([16])
                   'answer_cat',                    # torch.Size([16])          Label Index
                   'answer_cats',                   # torch.Size([16, 8864])    One-hot Label
                   'answer_cat_scores'])            # torch.Size([16, 8864])    One-hot Label

        """

        
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang_net(data_dict)        
        """
        dict_keys(['lang_out',                      # torch.Size([16, 13, 256])
                   'lang_emb',                      # torch.Size([16, 256])
                   'lang_mask',                     # torch.Size([16, 13])
        """
        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.detection_backbone(data_dict)
        """ torch.nonzero(data_dict["fp2_inds"].shape)
        dict_keys(['sa1_inds',      # torch.Size([16, 2048])
                   'sa1_xyz',       # torch.Size([16, 2048, 3])
                   'sa1_features',  # torch.Size([16, 128, 2048])
                   'sa2_inds',      # torch.Size([16, 1024])
                   'sa2_xyz',       # torch.Size([16, 1024, 3])
                   'sa2_features',  # torch.Size([16, 256, 1024])
                   'sa3_xyz',       # torch.Size([16, 512, 3])
                   'sa3_features',  # torch.Size([16, 256, 512])
                   'sa4_xyz',       # torch.Size([16, 256, 3])
                   'sa4_features',  # torch.Size([16, 256, 256])
                   'fp2_features',  # torch.Size([16, 256, 1024])
                   'fp2_xyz',       # torch.Size([16, 1024, 3])
                   'fp2_inds'       # torch.Size([16, 1024])
        """
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"] # batch_size, seed_feature_dim, num_seed, (16, 256, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz

        data_dict["seed_features"] = features
        xyz, features = self.voting_net(xyz, features) # batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        features_norm = torch.norm(features, p=2, dim=1) # torch.Size([16, 1024])
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        """ torch.nonzero(data_dict["vote_features"].shape)
        dict_keys:
                   'seed_inds',         # torch.Size([16, 1024])
                   'seed_xyz',          # torch.Size([16, 1024, 3])
                   'seed_features',     # torch.Size([16, 256, 1024])
                   'vote_xyz',          # torch.Size([16, 1024, 3])
                   'vote_features'      # torch.Size([16, 256, 1024])
        """

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal_net(xyz, features, data_dict)

        """ torch.nonzero(data_dict["bbox_sems"].shape)
        dict_keys:
                   'aggregated_vote_xyz'            # torch.Size([16, 256, 3])
                   'aggregated_vote_features'       # torch.Size([16, 256, 128])
                   'aggregated_vote_inds'           # torch.Size([16, 256])
                   'objectness_scores'              # torch.Size([16, 256, 2])
                   'center'                         # torch.Size([16, 256, 3])
                   'heading_scores'                 # torch.Size([16, 256, 1])
                   'heading_residuals_normalized'   # torch.Size([16, 256, 1])
                   'heading_residuals'              # torch.Size([16, 256, 1])
                   'size_scores'                    # torch.Size([16, 256, 18])      
                   'size_residuals_normalized'      # torch.Size([16, 256, 18, 3])
                   'size_residuals'                 # torch.Size([16, 256, 18, 3])
                   'sem_cls_scores'                 # torch.Size([16, 256, 18])
                   'bbox_corner'                    # torch.Size([16, 256, 8, 3])
                   'bbox_feature'                   # torch.Size([16, 256, 128])
                   'bbox_mask'                      # torch.Size([16, 256])
                   'bbox_sems'                      # torch.Size([16, 256])
        """

        #######################################
        #                                     #
        #             QA BACKBONE             #
        #                                     #
        #######################################

        # unpack outputs from question encoding branch
        lang_feat = data_dict["lang_out"] # word embeddings after LSTM (batch_size, num_words(max_question_length), hidden_size * num_dir)
        lang_mask = data_dict["lang_mask"] # word attetion (batch, num_words)
        # import pdb; pdb.set_trace()
        # unpack outputs from detection branch
        object_feat = data_dict['aggregated_vote_features'] # batch_size, num_proposal, proposal_size (128)
        if self.use_object_mask:
            object_mask = ~data_dict["bbox_mask"].bool().detach() #  # batch, num_proposals
        else:
            object_mask = None            

        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(1).unsqueeze(2)         # torch.Size([16, 1, 1, 13])
        if object_mask.dim() == 2:
            object_mask = object_mask.unsqueeze(1).unsqueeze(2)     # torch.Size([16, 1, 1, 256])   

        # --------- QA BACKBONE ---------
        # Pre-process Lanauge & Image Feature
        lang_feat = self.lang_feat_linear(lang_feat) # batch_size, num_words, hidden_size           torch.Size([16, 13, 256])
        object_feat = self.object_feat_linear(object_feat) # batch_size, num_proposal, hidden_size  torch.Size([16, 256, 256])

        # QA Backbone (Fusion network)
        lang_feat, object_feat = self.fusion_backbone(
            lang_feat,
            object_feat,
            lang_mask,
            object_mask,
        ) 
        # object_feat: batch_size, num_proposal, hidden_size    torch.Size([16, 256, 256])
        # lang_feat: batch_size, num_words, hidden_size         torch.Size([16, 13, 256])

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################
        if self.use_reference:   
            #  data_dict["cluster_ref"]:
            #  tensor([[-0.2910, -0.2910, -0.1096],
            #          [0.7795, -0.2910,  1.2384]]    
            # mask out invalid proposals
            object_conf_feat = object_feat * data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)
            data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1) 

        lang_feat = self.attflat_lang(          # torch.Size([16, 512])
                lang_feat,
                lang_mask
        )

        object_feat = self.attflat_visual(      # torch.Size([16, 512])
                object_feat,
                object_mask
        )

        fuse_feat = self.fusion_norm(lang_feat + object_feat) # batch, mcan_flat_out_size torch.Size([16, 512])

        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################
        if self.use_lang_cls:
            data_dict["lang_scores"] = self.lang_cls(fuse_feat) # batch_size, num_object_classe

        #######################################
        #                                     #
        #          QUESTION ANSERING          #
        #                                     #
        #######################################
        if self.use_answer:
            data_dict["answer_scores"] = self.answer_cls(fuse_feat) # batch_size, num_answers

        return data_dict
