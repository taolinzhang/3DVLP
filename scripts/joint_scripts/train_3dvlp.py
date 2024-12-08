# coding=utf8
# HACK ignore warnings
# fmt: off
import wandb
import numpy as np
import torch
import random
import argparse
import json
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
# sys.path.append(os.path.join('/data/zhangtaolin/data/ScanQa'))
from scripts.utils.script_utils import set_params_lr_dict
from scripts.utils.AdamW import AdamW
from models.jointnet.jointnet import JointNet
from lib.configs.config_joint import CONF
from lib.joint.solver_3dvlp import Solver
from lib.joint.dataset import ScannetReferenceDataset
from data.scannet.model_util_scannet import ScannetDatasetConfig
from copy import deepcopy
from datetime import datetime
from torch.utils.data import DataLoader
# fmt: on

# from crash_on_ipy import *

# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# extracted ScanNet object rotations from Scan2CAD
# NOTE some scenes are missing in this annotation!!!
SCANREFER_TRAIN = json.load(
    open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(
    open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))
SCAN2CAD_ROTATION = None

# constants
DC = ScannetDatasetConfig()

# import crash_on_ipy


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True,
                   scan2cad_rotation=None, batch_size=None, lang_num_max=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list,
        split=split,
        name=args.dataset,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=lang_num_max,
        lang_num_aug=args.lang_num_aug,
        augment=augment,
        shuffle=shuffle,
        scan2cad_rotation=scan2cad_rotation,
        use_distil=args.use_distil,
        minor_aug=args.minor_aug)
    if split == "train":
        should_shuffle = True
    else:
        should_shuffle = False
    print("should shuffle:", should_shuffle, "batch size", batch_size)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=should_shuffle, num_workers=4, prefetch_factor=4)

    return dataset, dataloader


def get_model(args, dataset, device):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * \
        3 + int(args.use_color) * 3 + int(not args.no_height)
    model = JointNet(num_class=DC.num_class,
                     vocabulary=None,
                     #  embeddings=dataset.glove,
                     num_heading_bin=DC.num_heading_bin,
                     num_size_cluster=DC.num_size_cluster,
                     mean_size_arr=DC.mean_size_arr,
                     input_feature_dim=input_channels,
                     num_proposal=args.num_proposals,
                     no_caption=args.no_caption,
                     use_topdown=args.use_topdown,
                     num_locals=args.num_locals,
                     query_mode=args.query_mode,
                     num_graph_steps=args.num_graph_steps,
                     use_relation=args.use_relation,
                     use_lang_classifier=(not args.no_lang_cls),
                     use_bidir=args.use_bidir,
                     no_reference=args.no_reference,
                     dataset_config=DC,
                     use_distil=args.use_distil,
                     unfreeze=args.unfreeze,
                     use_mlm=args.use_mlm,
                     use_con=args.use_con,
                     use_lang_emb=args.use_lang_emb,
                     mask_box=args.mask_box,
                     use_pc_encoder=args.use_pc_encoder,
                     use_match_con_loss=args.use_match_con_loss,
                     use_reg_head=args.use_reg_head,
                     use_kl_loss=args.use_kl_loss,
                     use_mlcv_net=args.use_mlcv_net,
                     use_vote_weight=args.use_vote_weight)

    if args.pretrain:
        print("loading pretrained VoteNet...")
        # pretrained_path = os.path.join(
        #     CONF.PATH.BASE, args.pretrain, "epoch_50.pth")
        pretrained_path = args.pretrain
        print("pretrained_path", pretrained_path, flush=True)
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    # multi-GPU
    if torch.cuda.device_count() > 1:
        print("using {} GPUs...".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # to device
    model.to(device)

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataset, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset["train"], device)

    weight_dict = {
        'lang': {'lr': 0.0005},
        'relation': {'lr': 0.0005},
        'match': {'lr': 0.0005},
        'caption': {'lr': 0.0005},
    }
    params = set_params_lr_dict(
        model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    # params = model.parameters()
    optimizer = AdamW(params, lr=args.lr,
                      weight_decay=args.wd, amsgrad=args.amsgrad)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    checkpoint_best = None

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag:
            stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)
        checkpoint = torch.load(
            os.path.join(args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_best = checkpoint["best"]
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag:
            stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_caption else None
    if args.coslr:
        if args.epoch > 200:
            LR_DECAY_STEP = {
                'type': 'cosine',
                'T_max': 200,
                'eta_min': 1e-5,
            }
        else:
            LR_DECAY_STEP = {
                'type': 'cosine',
                'T_max': args.epoch,
                'eta_min': 1e-5,
            }
    LR_DECAY_RATE = 0.1 if args.no_caption else None
    BN_DECAY_STEP = 20 if args.no_caption else None
    BN_DECAY_RATE = 0.5 if args.no_caption else None

    print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE,
          BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
    print("criterion", args.criterion, flush=True)
    solver = Solver(model=model,
                    args=args,
                    device=device,
                    config=DC,
                    dataset=dataset,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    stamp=stamp,
                    val_step=args.val_step,
                    num_ground_epoch=args.num_ground_epoch,
                    detection=not args.no_detection,
                    caption=not args.no_caption,
                    orientation=args.use_orientation,
                    distance=args.use_distance,
                    use_tf=args.use_tf,
                    reference=not args.no_reference,
                    use_lang_classifier=not args.no_lang_cls,
                    lr_decay_step=LR_DECAY_STEP,
                    lr_decay_rate=LR_DECAY_RATE,
                    bn_decay_step=BN_DECAY_STEP,
                    bn_decay_rate=BN_DECAY_RATE,
                    criterion=args.criterion,
                    checkpoint_best=checkpoint_best)
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted([
        line.rstrip() for line in open(
            os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))
    ])

    return scene_list


def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(
            open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(
            open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    if args.no_caption and args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(
            list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(
            list(set([data["scene_id"] for data in scanrefer_eval_val])))

        # filter data in chosen scenes
        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene
                       ) >= args.lang_num_max - args.lang_num_aug:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)
        scanrefer_train_new.append(scanrefer_train_new_scene)

        # 注意：new_scanrefer_eval_train实际上没用
        # eval on train
        new_scanrefer_eval_train = []
        scanrefer_eval_train_new = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)
            scanrefer_eval_train_new_scene = []
            for i in range(args.lang_num_max):
                scanrefer_eval_train_new_scene.append(data)
            scanrefer_eval_train_new.append(scanrefer_eval_train_new_scene)

        new_scanrefer_eval_val = scanrefer_eval_val
        scanrefer_eval_val_new = []
        scanrefer_eval_val_new_scene = []
        scene_id = ""
        for data in scanrefer_eval_val:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_eval_val_new_scene) > 0:
                    scanrefer_eval_val_new_scene.append(
                        scanrefer_eval_val_new_scene)
                scanrefer_eval_val_new_scene = []
            if len(scanrefer_eval_val_new_scene) >= args.lang_num_max:
                scanrefer_eval_val_new.append(scanrefer_eval_val_new_scene)
                scanrefer_eval_val_new_scene = []
            scanrefer_eval_val_new_scene.append(data)
        scanrefer_eval_val_new.append(scanrefer_eval_val_new_scene)

        new_scanrefer_eval_val2 = []
        scanrefer_eval_val_new2 = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val2.append(data)
            scanrefer_eval_val_new_scene2 = []
            for i in range(args.lang_num_max):
                scanrefer_eval_val_new_scene2.append(data)
            scanrefer_eval_val_new2.append(scanrefer_eval_val_new_scene2)

        val_lang_num_max = 1
        eval_ground_val = scanrefer_eval_val
        eval_ground_new = []
        eval_ground_new_scene = []
        scene_id = ""
        for data in scanrefer_eval_val:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(eval_ground_new_scene) > 0:
                    eval_ground_new_scene.append(
                        eval_ground_new_scene)
                eval_ground_new_scene = []
            if len(eval_ground_new_scene) >= val_lang_num_max:
                eval_ground_new.append(eval_ground_new_scene)
                eval_ground_new_scene = []
            eval_ground_new_scene.append(data)
        eval_ground_new.append(eval_ground_new_scene)

    print("scanrefer_train_new", len(scanrefer_train_new),
          len(scanrefer_train_new[0]))
    print("scanrefer_eval_new", len(scanrefer_eval_train_new),
          len(scanrefer_eval_val_new))
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    print("sum", sum)  # 1418 363

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(args.dataset))
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train),
                                                      len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(
        len(new_scanrefer_eval_train), len(new_scanrefer_eval_val)))
    print(f"ground_eval secens: {len(eval_ground_val)}")

    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, new_scanrefer_eval_val2, eval_ground_val, all_scene_list, scanrefer_train_new, scanrefer_eval_train_new, scanrefer_eval_val_new, scanrefer_eval_val_new2, eval_ground_new


def get_scanrefer_test(args):
    SCANREFER_TEST = json.load(
        open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))
    scanrefer = SCANREFER_TEST
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    new_scanrefer_val = scanrefer
    scanrefer_val_new = []
    scanrefer_val_new_scene = []
    scene_id = ""
    for data in scanrefer:
        # if data["scene_id"] not in scanrefer_val_new:
        # scanrefer_val_new[data["scene_id"]] = []
        # scanrefer_val_new[data["scene_id"]].append(data)
        if scene_id != data["scene_id"]:
            scene_id = data["scene_id"]
            if len(scanrefer_val_new_scene) > 0:
                scanrefer_val_new.append(scanrefer_val_new_scene)
            scanrefer_val_new_scene = []
        if len(scanrefer_val_new_scene) >= args.lang_num_max:
            scanrefer_val_new.append(scanrefer_val_new_scene)
            scanrefer_val_new_scene = []
        scanrefer_val_new_scene.append(data)
    if len(scanrefer_val_new_scene) > 0:
        scanrefer_val_new.append(scanrefer_val_new_scene)

    return scanrefer, scene_list, scanrefer_val_new


def predict(args):
    from tqdm import tqdm
    from lib.ap_helper.ap_helper_fcos import parse_predictions
    from utils.box_util import get_3d_box
    print("predict bounding boxes...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list, scanrefer_val_new = get_scanrefer_test(args)

    # dataloader
    #_, dataloader = get_dataloader(args, scanrefer, scene_list, "test", DC)
    dataset, dataloader = get_dataloader(args, scanrefer, scanrefer_val_new, scene_list, "test",
                                         DC, False, shuffle=False, batch_size=args.batch_size, lang_num_max=1)

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset, device)

    # config
    # POST_DICT = {
    #     "remove_empty_box": True,
    #     "use_3d_nms": True,
    #     "nms_iou": 0.25,
    #     "use_old_type_nms": False,
    #     "cls_nms": True,
    #     "per_class_proposal": True,
    #     "conf_thresh": 0.05,
    #     "dataset_config": DC
    # } if not args.no_nms else None

    # predict
    print("predicting...")
    pred_bboxes = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        # feed
        with torch.no_grad():
            data_dict = model(data_dict)
            """
            _, data_dict = get_loss(
                data_dict=data_dict, 
                config=DC, 
                detection=False,
                reference=True
            )
            """

            objectness_preds_batch = torch.argmax(
                data_dict['objectness_scores'], 2).long()

            # if POST_DICT:
            #     _ = parse_predictions(data_dict, POST_DICT)
            #     nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

            #     # construct valid mask
            #     pred_masks = (nms_masks * objectness_preds_batch == 1).float()
            # else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

            # pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
            pred_ref = torch.argmax(
                data_dict['cluster_ref'], 1)  # (B*lang_num_max,)
            batch_size, lang_num_max = data_dict["input_ids"].shape[:2]
            pred_ref = pred_ref.view(batch_size, lang_num_max).cpu().numpy()
            # pred_center = data_dict['center'] # (B,K,3)
            pred_heading = data_dict['pred_heading'].detach(
            ).cpu().numpy()  # B,num_proposal
            pred_center = data_dict['pred_center'].detach(
            ).cpu().numpy()  # (B, num_proposal)
            pred_box_size = data_dict['pred_size'].detach(
            ).cpu().numpy()  # (B, num_proposal, 3)

            for i in range(pred_ref.shape[0]):
                # compute the iou
                pred_ref_idx = pred_ref[i]
                pred_center_ids = pred_center[i][pred_ref_idx][0]
                pred_heading_ids = pred_heading[i][pred_ref_idx].item()
                pred_box_size_ids = pred_box_size[i][pred_ref_idx][0]
                # from IPython import embed
                # embed()
                pred_bbox = get_3d_box(
                    pred_box_size_ids, pred_heading_ids, pred_center_ids)

                # construct the multiple mask
                #multiple = data_dict["unique_multiple"][i].item()
                multiple = data_dict["unique_multiple_list"][i][0].item()

                # construct the others mask
                #others = 1 if data_dict["object_cat"][i] == 17 else 0
                others = 1 if data_dict["object_cat_list"][i][0] == 17 else 0

                # store data
                scanrefer_idx = data_dict["scan_idx"][i].item()
                pred_data = {
                    "scene_id": scanrefer[scanrefer_idx]["scene_id"],
                    "object_id": scanrefer[scanrefer_idx]["object_id"],
                    "ann_id": scanrefer[scanrefer_idx]["ann_id"],
                    "bbox": pred_bbox.tolist(),
                    "unique_multiple": multiple,
                    "others": others
                }
                pred_bboxes.append(pred_data)

    # dump
    print("dumping...")
    pred_path = os.path.join("./pred.json")
    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, scanrefer_eval_val2, eval_ground_val, all_scene_list, scanrefer_train_new, scanrefer_eval_train_new, scanrefer_eval_val_new, scanrefer_eval_val_new2, eval_ground_new = get_scanrefer(
        args)

    # 注意：eval_train_dataset实际上没用
    # dataloader
    train_dataset, train_dataloader = get_dataloader(
        args, scanrefer_train, scanrefer_train_new, all_scene_list, "train", DC, True, SCAN2CAD_ROTATION, batch_size=args.batch_size, lang_num_max=args.lang_num_max)
    eval_train_dataset, eval_train_dataloader = get_dataloader(
        args, scanrefer_eval_train, scanrefer_eval_train_new, all_scene_list, "val", DC, False, shuffle=False, batch_size=args.batch_size, lang_num_max=args.lang_num_max)
    eval_val_dataset, eval_val_dataloader = get_dataloader(
        args, scanrefer_eval_val, scanrefer_eval_val_new, all_scene_list, "val", DC, False, shuffle=False, batch_size=args.batch_size, lang_num_max=args.lang_num_max)
    eval_val_dataset2, eval_val_dataloader2 = get_dataloader(
        args, scanrefer_eval_val2, scanrefer_eval_val_new2, all_scene_list, "val", DC, False, shuffle=False, batch_size=args.batch_size, lang_num_max=args.lang_num_max)
    eval_ground_dataset, eval_ground_dataloader = get_dataloader(
        args, eval_ground_val, eval_ground_new, all_scene_list, "val", DC, False, shuffle=False, batch_size=6, lang_num_max=1
    )

    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset,
            "val_scene": eval_val_dataset2,
            "ground": eval_ground_dataset,
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader,
            "val_scene": eval_val_dataloader2,
            "ground": eval_ground_dataloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",
                        type=str,
                        help="tag for the training, e.g. cuda_wl",
                        default="")
    parser.add_argument("--dataset",
                        type=str,
                        help="Choose a dataset: ScanRefer or ReferIt3D",
                        default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--epoch", type=int,
                        help="number of epochs", default=20)
    parser.add_argument("--verbose",
                        type=int,
                        help="iterations of showing verbose",
                        default=10)
    parser.add_argument("--val_step",
                        type=int,
                        help="iterations of validating",
                        default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-3)
    parser.add_argument("--amsgrad", action='store_true',
                        help="optimizer with amsgrad")

    parser.add_argument("--lang_num_max", type=int,
                        help="lang num max", default=8)
    parser.add_argument("--num_points",
                        type=int,
                        default=40000,
                        help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals",
                        type=int,
                        default=256,
                        help="Proposal number [default: 256]")
    parser.add_argument("--num_locals",
                        type=int,
                        default=-1,
                        help="Number of local objects [default: -1]")
    parser.add_argument("--num_scenes",
                        type=int,
                        default=-1,
                        help="Number of scenes [default: -1]")
    parser.add_argument("--num_graph_steps",
                        type=int,
                        default=0,
                        help="Number of graph conv layer [default: 0]")
    parser.add_argument("--num_ground_epoch",
                        type=int,
                        default=50,
                        help="Number of ground epoch [default: 50]")

    parser.add_argument(
        "--criterion",
        type=str,
        default="sum",
        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]"
    )

    parser.add_argument(
        "--query_mode",
        type=str,
        default="center",
        help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument(
        "--graph_mode",
        type=str,
        default="edge_conv",
        help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr",
                        type=str,
                        default="add",
                        help="Mode for aggregating features, [choices: add, mean, max]")

    parser.add_argument("--coslr", action='store_true',
                        help="cosine learning rate")
    parser.add_argument("--no_height",
                        action="store_true",
                        help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment",
                        action="store_true",
                        help="Do NOT use height signal in input.")
    parser.add_argument("--no_detection",
                        action="store_true",
                        help="Do NOT train the detection module.")
    parser.add_argument("--no_caption",
                        action="store_true",
                        help="Do NOT train the caption module.")
    parser.add_argument("--no_lang_cls",
                        action="store_true",
                        help="Do NOT use language classifier.")
    parser.add_argument("--no_reference",
                        action="store_true",
                        help="Do NOT train the localization module.")

    parser.add_argument("--use_tf",
                        action="store_true",
                        help="enable teacher forcing in inference.")
    parser.add_argument("--use_color",
                        action="store_true",
                        help="Use RGB color in input.")
    parser.add_argument("--use_normal",
                        action="store_true",
                        help="Use RGB color in input.")
    parser.add_argument("--use_multiview",
                        action="store_true",
                        help="Use multiview images.")
    parser.add_argument("--use_topdown",
                        action="store_true",
                        help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation",
                        action="store_true",
                        help="Use object-to-object relation in graph.")
    parser.add_argument("--use_new", action="store_true",
                        help="Use new Top-down module.")
    parser.add_argument("--use_orientation",
                        action="store_true",
                        help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--use_distance",
                        action="store_true",
                        help="Use object-to-object distance loss in graph.")
    parser.add_argument("--use_bidir",
                        action="store_true",
                        help="Use bi-directional GRU.")
    parser.add_argument(
        "--pretrain",
        type=str,
        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint",
                        type=str,
                        help="Specify the checkpoint root",
                        default="")

    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    parser.add_argument("--use_distil", action="store_true",
                        help="Use DistilBert model")
    parser.add_argument("--use_con", action="store_true",
                        help="Use constrastive module")
    parser.add_argument("--use_mlm", action="store_true",
                        help="Use MLM pretrain task")
    parser.add_argument("--lang_num_aug",
                        type=int,
                        help="Augmented samples in lang num max",
                        default=0)
    parser.add_argument("--unfreeze",
                        type=int,
                        help="Unfreeze layers for Bert",
                        default=6)
    parser.add_argument("--use_lang_emb",
                        action="store_true",
                        help="Use lang_embed for cross attention")
    parser.add_argument("--mask_box",
                        action="store_true",
                        help="Use mask box augmentation")
    parser.add_argument("--use_pc_encoder",
                        action="store_true",
                        help="Use pointpollars as point cloud encoder")
    parser.add_argument("--use_match_con_loss",
                        action="store_true",
                        help="Use con loss for match module")
    parser.add_argument("--use_diou_loss",
                        action="store_true", help="Use diou loss")
    parser.add_argument("--use_attr_loss",
                        action="store_true", help="Use attr loss")
    parser.add_argument("--use_vote_weight",
                        action="store_true",
                        help="Use vote weight loss.")
    parser.add_argument("--use_reg_head",
                        action="store_true",
                        help="Use box regression head")
    parser.add_argument("--use_kl_loss",
                        action="store_true",
                        help="Use kl loss for box regression")
    parser.add_argument(
        "--use_answer", action="store_true", help="Use answer.")
    parser.add_argument(
        "--use_mlcv_net", action="store_true", help="Use mlcv net.")
    parser.add_argument("--minor_aug",
                        action="store_true",
                        help="augamation for minor label")

    args = parser.parse_args()

    # # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project="3dvlp", entity="3dvlp", name="3dvlp")
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("iter")
    wandb.define_metric("iter/*", step_metric="iter")
    train(args)
