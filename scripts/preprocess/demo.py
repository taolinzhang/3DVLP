import os
import sys
import json
import h5py
from tqdm import tqdm
import numpy as np
import time

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.configs.config import CONF
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(
        CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def work(split, use_color=False, use_normal=True, use_multiview=True):
    # multiview_data = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
    scene_list = get_scannet_scene_list(split)

    start_time = time.time()

    for scene_id in tqdm(scene_list[:1000]):
        # load scene data

        np.load("/dev/shm/metadata/scannet_data/scene0000_00_pcl_color_train.npy")

        np.load("/dev/shm/metadata/scannet_data/scene0000_00_preprocess_train.npy")

    print(time.time() - start_time)

if __name__ == "__main__":
    print("preprocess train dataset")
    work('train', use_color=False, use_normal=True, use_multiview=True)
    print("preprocess val dataset")
    work('val', use_color=False, use_normal=True, use_multiview=True)
