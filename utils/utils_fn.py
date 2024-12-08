import torch
import numpy as np
from data.scannet.model_util_scannet import rotate_aligned_boxes_along_axis
from utils.pc_utils import rotx, roty, rotz
import os
import json
import time


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def debug():
    assert(False)


def flip_augment(point_cloud, target_bboxes, rng):
    """ Apply flip augment to the point cloud and target boxes"""
    if rng.random() > 0.7:
        # Flipping along the YZ plane
        point_cloud[:, 0] = -1 * point_cloud[:, 0]
        target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

    if rng.random() > 0.7:
        # Flipping along the XZ plane
        point_cloud[:, 1] = -1 * point_cloud[:, 1]
        target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
    return point_cloud, target_bboxes


# def rotate_augment(point_cloud, target_bboxes, rng):
#     """ Apply rotation augment to the point cloud and target boxes"""
#     # Rotation along X-axis
#     start = time.time()
#     rot_angle = (rng.random()*np.pi/18) - \
#         np.pi/36  # -5 ~ +5 degree
#     print("2", time.time() - start)
#     rot_mat = rotx(rot_angle)
#     print("1", time.time() - start)
#     point_cloud[:, 0:3] = np.dot(
#         point_cloud[:, 0:3], np.transpose(rot_mat))
#     print("1.1", time.time() - start)
#     target_bboxes = rotate_aligned_boxes_along_axis(
#         target_bboxes, rot_mat, "x")
#     print("3", time.time() - start)
#     # Rotation along Y-axis
#     rot_angle = (rng.random()*np.pi/18) - \
#         np.pi/36  # -5 ~ +5 degree
#     rot_mat = roty(rot_angle)
#     print("4", time.time() - start)
#     point_cloud[:, 0:3] = np.dot(
#         point_cloud[:, 0:3], np.transpose(rot_mat))
#     target_bboxes = rotate_aligned_boxes_along_axis(
#         target_bboxes, rot_mat, "y")
#     print("5", time.time() - start)
#     # Rotation along up-axis/Z-axis
#     rot_angle = (rng.random()*np.pi/18) - \
#         np.pi/36  # -5 ~ +5 degree

#     rot_mat = rotz(rot_angle)
#     print("6", time.time() - start)
#     point_cloud[:, 0:3] = np.dot(
#         point_cloud[:, 0:3], np.transpose(rot_mat))
#     target_bboxes = rotate_aligned_boxes_along_axis(
#         target_bboxes, rot_mat, "z")
#     print("7", time.time() - start)
#     return point_cloud, target_bboxes

def rotate_augment(point_cloud, target_bboxes, rng):
    """ Apply rotation augment to the point cloud and target boxes"""
    # Rotation along X-axis
    rot_angle = (rng.random()*np.pi/18) - np.pi/36  # -5 ~ +5 degree
    rot_mat_x = rotx(rot_angle)
    target_bboxes = rotate_aligned_boxes_along_axis(
        target_bboxes, rot_mat_x, "x")

    # Rotation along Y-axis
    rot_angle = (rng.random()*np.pi/18) - np.pi/36  # -5 ~ +5 degree
    rot_mat_y = roty(rot_angle)
    target_bboxes = rotate_aligned_boxes_along_axis(
        target_bboxes, rot_mat_y, "y")

    # Rotation along up-axis/Z-axis
    rot_angle = (rng.random()*np.pi/18) - np.pi/36  # -5 ~ +5 degree
    rot_mat_z = rotz(rot_angle)
    target_bboxes = rotate_aligned_boxes_along_axis(
        target_bboxes, rot_mat_z, "z")

    # Rotate Point_Cloud
    rot_mat = np.dot(np.transpose(rot_mat_x), np.transpose(rot_mat_y))
    rot_mat = np.dot(rot_mat, np.transpose(rot_mat_z))
    point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], rot_mat)

    return point_cloud, target_bboxes


def scale_augment(point_cloud, target_bboxes, use_height, rng):
    """ Apply scale augment to the point cloud and target boxes"""
    # print('Warning! Dont Use Extra Augmentation!(votenet didnot use it)', flush=True)
    # NEW: scale from 0.8 to 1.2
    scale = rng.uniform(-0.1, 0.1, (3, 3))
    scale = np.exp(scale)
    scale = scale * np.eye(3)
    point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
    if use_height:
        point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
    target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
    target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)

    # Translation
    point_cloud, target_bboxes = translate(
        point_cloud, target_bboxes, rng)
    return point_cloud, target_bboxes


def translate(point_set, bbox, rng):
    # unpack
    coords = point_set[:, :3]

    # translation factors
    x_factor = rng.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
    y_factor = rng.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
    z_factor = rng.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
    factor = [x_factor, y_factor, z_factor]

    # dump
    coords += factor
    point_set[:, :3] = coords
    bbox[:, :3] += factor

    return point_set, bbox


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return num_params


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


def final_eval_fn(masks, others, ref_acc, ious, lang_acc):
    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(
                masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)

    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >=
                                     0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))