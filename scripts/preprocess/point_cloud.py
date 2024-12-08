import os
import sys
import json
import h5py
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.configs.config import CONF
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(
        CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def work(split, use_color=False, use_normal=True, use_multiview=True):
    multiview_data = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
    scene_list = get_scannet_scene_list(split)

    dump_data = []

    for scene_id in tqdm(scene_list[:10]):
        # load scene data
        mesh_vertices = np.load(os.path.join(
            CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy")

        # use color
        if not use_color:
            if not use_color:
                point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
                pcl_color = mesh_vertices[:, 3:6]
            else:
                point_cloud = mesh_vertices[:, 0:6]
                point_cloud[:, 3:6] = (
                    point_cloud[:, 3:6]-MEAN_COLOR_RGB)/256.0
                pcl_color = point_cloud[:, 3:6]

            if use_normal:
                normals = mesh_vertices[:, 6:9]
                point_cloud = np.concatenate([point_cloud, normals], 1)

            if use_multiview:
                multiview = multiview_data[scene_id]
                point_cloud = np.concatenate([point_cloud, multiview], 1)
            
            dump_data.append(point_cloud.tolist())

            np.save(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) +
                    "_preprocess_{}.npy".format(split), point_cloud)

            np.save(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) +
                    "_pcl_color_{}.npy".format(split), pcl_color)
    with open("dump.json", "w") as f:
        json.dump(dump_data, f)

if __name__ == "__main__":
    print("preprocess train dataset")
    work('train', use_color=False, use_normal=True, use_multiview=True)
    print("preprocess val dataset")
    work('val', use_color=False, use_normal=True, use_multiview=True)
