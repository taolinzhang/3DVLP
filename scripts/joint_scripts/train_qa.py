# fmt: off
import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
import wandb
import random

from torch.utils.data import DataLoader
from datetime import datetime

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.vqa.dataset import ScannetQADataset, ScannetQADatasetConfig
from lib.vqa.solver import Solver
from lib.configs.config_vqa import CONF 
from models.jointnet.jointnet import JointNet
# fmt: on
project_name = "ScanQA_v1.0"
SCANQA_TRAIN = json.load(
    open(os.path.join(CONF.PATH.SCANQA, project_name + "_train.json")))
SCANQA_VAL = json.load(
    open(os.path.join(CONF.PATH.SCANQA, project_name + "_val.json")))

# constants
DC = ScannetQADatasetConfig()


def get_answer_cands(args, scanqa):
    answer_counter = sum([data["answers"] for data in scanqa["train"]], [])
    answer_counter = collections.Counter(sorted(answer_counter))
    num_all_answers = len(answer_counter)
    answer_max_size = args.answer_max_size
    if answer_max_size < 0:
        answer_max_size = len(answer_counter)
    answer_counter = dict([x for x in answer_counter.most_common()[
                          :answer_max_size] if x[1] >= args.answer_min_freq])
    print("using {} answers out of {} ones".format(
        len(answer_counter), num_all_answers))
    answer_cands = sorted(answer_counter.keys())
    return answer_cands, answer_counter


def get_dataloader(args, scanqa, all_scene_list, split, config, augment):
    answer_cands, answer_counter = get_answer_cands(args, scanqa)
    config.num_answers = len(answer_cands)

    if 'bert-' in args.tokenizer_name:
        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None

    dataset = ScannetQADataset(
        scanqa=scanqa[split],
        scanqa_all_scene=all_scene_list,
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        tokenizer=tokenizer,
        augment=augment,
        debug=args.debug,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader


def get_model(args, config):
    if "bert-" in args.tokenizer_name:
        from transformers import AutoConfig
        bert_model_name = args.tokenizer_name
        bert_config = AutoConfig.from_pretrained(bert_model_name)
        if hasattr(bert_config, "hidden_size"):
            lang_emb_size = bert_config.hidden_size
        else:
            # for distllbert
            lang_emb_size = bert_config.dim
    else:
        bert_model_name = None
        lang_emb_size = 300  # glove emb_size

    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * \
        3 + int(args.use_color) * 3 + int(not args.no_height)

    model = JointNet(
        num_class=DC.num_class,
        vocabulary=None,
        embeddings=None,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=True,
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
        use_answer=True,
        num_answers=config.num_answers,)

    if args.pretrain:
        pretrained_path = os.path.join(
            CONF.PATH.BASE, args.pretrain, "model_last.pth")
        print("pretrained_path", pretrained_path, flush=True)
        pretrained_dict = torch.load(pretrained_path)
        model.load_state_dict(pretrained_dict, strict=False)
    # to CUDA
    model = model.cuda()
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataloader):
    model = get_model(args, DC)
    #wandb.watch(model, log_freq=100)

    if args.optim_name == 'adam':
        model_params = [{"params": model.parameters()}]
        optimizer = optim.Adam(
            model_params,
            lr=args.lr,
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            weight_decay=args.wd,
            amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd,
                                amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw_cb':
        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          betas=[args.adam_beta1, args.adam_beta2],
                          eps=args.adam_epsilon,
                          weight_decay=args.wd)
    else:
        raise NotImplementedError

    print('set optimizer...')
    print(optimizer)
    print()

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(
            CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag:
            stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    loss_weights = {}
    loss_weights['vote_loss'] = args.vote_loss_weight
    loss_weights['objectness_loss'] = args.objectness_loss_weight
    loss_weights['box_loss'] = args.box_loss_weight
    loss_weights['sem_cls_loss'] = args.sem_cls_loss_weight
    loss_weights['ref_loss'] = args.ref_loss_weight
    loss_weights['lang_loss'] = args.lang_loss_weight
    loss_weights['answer_loss'] = args.answer_loss_weight

    solver = Solver(
        args=args,
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        cur_criterion=args.cur_criterion,
        detection=not args.no_detection,
        use_reference=not args.no_reference,
        use_answer=True,
        use_lang_classifier=not args.no_lang_cls,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
        loss_weights=loss_weights,
    )
    num_params = get_num_params(model)

    return solver, num_params, root, stamp


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    answer_vocab = train_dataset.answer_counter
    with open(os.path.join(root, "answer_vocab.json"), "w") as f:
        json.dump(answer_vocab, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(
        CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def get_scanqa(scanqa_train, scanqa_val, train_num_scenes, val_num_scenes):
    # get initial scene list
    train_scene_list = sorted(
        list(set([data["scene_id"] for data in scanqa_train])))
    val_scene_list = sorted(
        list(set([data["scene_id"] for data in scanqa_val])))

    # set train_num_scenes
    if train_num_scenes <= -1:
        train_num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= train_num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:train_num_scenes]

    # filter data in chosen scenes
    new_scanqa_train = []
    for data in scanqa_train:
        if data["scene_id"] in train_scene_list:
            new_scanqa_train.append(data)

    # set val_num_scenes
    if val_num_scenes <= -1:
        val_num_scenes = len(val_scene_list)
    else:
        assert len(val_scene_list) >= val_num_scenes

    # slice val_scene_list
    val_scene_list = val_scene_list[:val_num_scenes]

    new_scanqa_val = []
    for data in scanqa_val:
        if data["scene_id"] in val_scene_list:
            new_scanqa_val.append(data)

    # new_scanqa_val = scanqa_val[0:4] # debugging

    # all scanqa scene
    all_scene_list = train_scene_list + val_scene_list
    #print("train on {} samples and val on {} samples".format(len(new_scanqa_train), len(new_scanqa_val)))
    # exit()
    return new_scanqa_train, new_scanqa_val, all_scene_list


def train(args):
    # WandB init
    # wandb.init(project=project_name, config=args)

    # init training dataset
    print("preparing data...")
    scanqa_train, scanqa_val, all_scene_list = get_scanqa(
        SCANQA_TRAIN, SCANQA_VAL, args.train_num_scenes, args.val_num_scenes)
    scanqa = {
        "train": scanqa_train,
        "val": scanqa_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(
        args, scanqa, all_scene_list, "train", DC, not args.no_augment)
    val_dataset, val_dataloader = get_dataloader(
        args, scanqa, all_scene_list, "val", DC, False)
    print("train on {} samples and val on {} samples".format(
        len(train_dataset), len(val_dataset)))

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root, stamp = get_solver(args, dataloader)
    if stamp:
        wandb.run.name = stamp
        wandb.run.save()

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str,
                        help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--dataset", type=str,
                        help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")

    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--epoch", type=int,
                        help="number of epochs", default=20)
    parser.add_argument("--verbose", type=int,
                        help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int,
                        help="iterations of validating", default=2000)

    parser.add_argument("--lang_num_max", type=int,
                        help="lang num max", default=8)
    parser.add_argument("--num_locals", type=int, default=-1,
                        help="Number of local objects [default: -1]")
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="Number of scenes [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0,
                        help="Number of graph conv layer [default: 0]")
    parser.add_argument("--num_ground_epoch", type=int,
                        default=50, help="Number of ground epoch [default: 50]")

    parser.add_argument("--criterion", type=str, default="sum",
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")

    parser.add_argument("--query_mode", type=str, default="center",
                        help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv",
                        help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add",
                        help="Mode for aggregating features, [choices: add, mean, max]")

    parser.add_argument("--coslr", action='store_true',
                        help="cosine learning rate")
    parser.add_argument("--no_detection", action="store_true",
                        help="Do NOT train the detection module.")
    parser.add_argument("--no_caption", action="store_true",
                        help="Do NOT train the caption module.")
    parser.add_argument("--no_reference", action="store_true",
                        help="Do NOT train the localization module.")

    parser.add_argument("--use_tf", action="store_true",
                        help="enable teacher forcing in inference.")
    parser.add_argument("--use_topdown", action="store_true",
                        help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true",
                        help="Use object-to-object relation in graph.")
    parser.add_argument("--use_new", action="store_true",
                        help="Use new Top-down module.")
    parser.add_argument("--use_orientation", action="store_true",
                        help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--use_distance", action="store_true",
                        help="Use object-to-object distance loss in graph.")
    parser.add_argument("--use_bidir", action="store_true",
                        help="Use bi-directional GRU.")
    parser.add_argument("--pretrain", type=str,
                        help="Specify the folder name containing the pretrained detection module.")

    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    parser.add_argument("--use_distil", action="store_true",
                        help="Use DistilBert model")
    parser.add_argument("--use_con", action="store_true",
                        help="Use constrastive module")
    parser.add_argument("--use_mlm", action="store_true",
                        help="Use MLM pretrain task")
    parser.add_argument("--lang_num_aug", type=int,
                        help="Augmented samples in lang num max", default=0)
    parser.add_argument("--unfreeze", type=int,
                        help="Unfreeze layers for Bert", default=6)
    parser.add_argument("--use_lang_emb", action="store_true",
                        help="Use lang_embed for cross attention")
    parser.add_argument("--mask_box", action="store_true",
                        help="Use mask box augmentation")
    parser.add_argument("--use_pc_encoder", action="store_true",
                        help="Use pointpollars as point cloud encoder")
    parser.add_argument("--use_match_con_loss", action="store_true",
                        help="Use con loss for match module")
    parser.add_argument("--use_diou_loss", action="store_true",
                        help="Use diou loss")
    parser.add_argument("--use_reg_head", action="store_true",
                        help="Use box regression head")
    parser.add_argument("--use_kl_loss", action="store_true",
                        help="Use kl loss for box regression")

    parser.add_argument("--cur_criterion", type=str,
                        default="answer_acc_at1", help="data augmentation type")
    parser.add_argument("--train_num_scenes", type=int,
                        default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int,
                        default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Optimizer
    parser.add_argument("--optim_name", type=str,
                        help="optimizer name", default="adam")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lr", type=float,
                        help="initial learning rate", default=5e-4)
    parser.add_argument("--adam_beta1", type=float,
                        help="beta1 hyperparameter for the Adam optimizer", default=0.9)
    parser.add_argument("--adam_beta2", type=float,
                        help="beta2 hyperparameter for the Adam optimizer", default=0.999)  # 0.98
    parser.add_argument("--adam_epsilon", type=float,
                        help="epsilon hyperparameter for the Adam optimizer", default=1e-8)  # 1e-9
    parser.add_argument("--amsgrad", action="store_true",
                        help="Use amsgrad for Adam")
    parser.add_argument('--lr_decay_step', nargs='+',
                        type=int, default=[100, 200])  # 15
    parser.add_argument("--lr_decay_rate", type=float,
                        help="decay rate of learning rate", default=0.2)  # 01, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float,
                        help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float,
                        help="Maximum gradient norm ", default=1.0)
    # Data
    parser.add_argument("--num_points", type=int, default=40000,
                        help="Point Number [default: 40000]")
    parser.add_argument("--no_height", action="store_true",
                        help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true",
                        help="Do NOT use data augmentations.")
    parser.add_argument("--use_color", action="store_true",
                        help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true",
                        help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true",
                        help="Use multiview images.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden layer size[default: 256]")
    ## pointnet & votenet & proposal
    parser.add_argument("--vote_radius", type=float, help="", default=0.3)  # 5
    parser.add_argument("--vote_nsample", type=int, help="", default=16)  # 512
    parser.add_argument("--pointnet_width", type=int, help="", default=1)
    parser.add_argument("--pointnet_depth", type=int, help="", default=2)
    parser.add_argument("--seed_feat_dim", type=int,
                        help="", default=256)  # or 288
    parser.add_argument("--proposal_size", type=int, help="", default=128)
    parser.add_argument("--num_proposals", type=int,
                        default=256, help="Proposal number [default: 256]")
    parser.add_argument("--use_seed_lang", action="store_true",
                        help="Fuse seed feature and language feature.")
    # module option
    parser.add_argument("--no_object_mask", action="store_true",
                        help="objectness_mask for qa")
    parser.add_argument("--no_lang_cls", action="store_true",
                        help="Do NOT use language classifier.")

    # Pretrain
    parser.add_argument("--use_checkpoint", type=str,
                        help="Specify the checkpoint root", default="")
    # Loss
    parser.add_argument("--vote_loss_weight", type=float,
                        help="vote_net loss weight", default=1.0)
    parser.add_argument("--objectness_loss_weight", type=float,
                        help="objectness loss weight", default=0.5)
    parser.add_argument("--box_loss_weight", type=float,
                        help="box loss weight", default=1.0)
    parser.add_argument("--sem_cls_loss_weight", type=float,
                        help="sem_cls loss weight", default=0.1)
    parser.add_argument("--ref_loss_weight", type=float,
                        help="reference loss weight", default=0.1)
    parser.add_argument("--lang_loss_weight", type=float,
                        help="language loss weight", default=0.1)
    parser.add_argument("--answer_loss_weight", type=float,
                        help="answer loss weight", default=0.1)
    # Answer
    parser.add_argument("--answer_cls_loss", type=str,
                        help="answer classifier loss", default="bce")  # ce, bce
    parser.add_argument("--answer_max_size", type=int,
                        help="maximum size of answer candicates", default=-1)  # default use all
    parser.add_argument("--answer_min_freq", type=int,
                        help="minimum frequence of answers", default=1)
    parser.add_argument("--answer_pdrop", type=float,
                        help="dropout_rate of answer_cls", default=0.3)
    # Question
    # or bert-base-uncased, bert-large-uncased-whole-word-masking, distilbert-base-uncased
    parser.add_argument("--tokenizer_name", type=str,
                        help="Pretrained tokenizer name", default="bert-base-uncased")
    parser.add_argument("--lang_num_layers", type=int,
                        default=1, help="Number of GRU layers")
    parser.add_argument("--lang_use_bidir", action="store_true",
                        help="Use bi-directional GRU.")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze BERT ebmedding model")
    parser.add_argument("--finetune_bert_last_layer",
                        action="store_true", help="Finetue BERT last layer")
    parser.add_argument("--use_answer", action="store_true",
                        help="Use answer.")

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

    wandb.init(project="A100", entity="3dplp",
               name="3djcg_qa_pretrain")
    # wandb.init()
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("iter")
    wandb.define_metric("iter/*", step_metric="iter")
    train(args)
