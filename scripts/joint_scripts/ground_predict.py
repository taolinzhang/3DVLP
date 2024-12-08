import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.configs.config_joint import CONF
from lib.joint.dataset import ScannetReferenceDataset
from lib.joint.solver import Solver
from lib.ap_helper.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper.loss_joint import get_joint_loss
from lib.joint.eval_ground import get_eval
from utils.box_util import get_3d_box
from models.jointnet.jointnet import JointNet
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANREFER_TEST = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))
# SCANREFER_TEST = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list,
        split=split,
        name=args.dataset,
        num_points=args.num_points,
        use_color=args.use_color,
        use_height=(not args.no_height),
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max
    )
    print("predict for {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, dataloader

def get_model(args, DC, dataset):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = JointNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=True,
        use_topdown=False,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        dataset_config=DC
    ).cuda()

    model_name = "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    load_result = model.load_state_dict(torch.load(path), strict=False)
    print(load_result, 'loading done', flush=True)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=1)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    args = parser.parse_args()

    assert args.lang_num_max == 1, 'Test: Lang Num Max == 1'
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    predict(args)
