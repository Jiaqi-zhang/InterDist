import os
import random
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

import torch
from torch.utils import data
from utils.preprocess import load_interhuman_motion
from utils.quaternion import qmul_np, qinv_np, qrot_np
from utils.utils import process_motion_np, rigid_transform, MotionNormalizer


def preprocess_motion(motions):
    motion1, motion2 = motions
    motion1_reset, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
    motion2_reset, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)

    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
    xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]
    
    # rigid_transform returns a reference value, so a deep copy is needed here.
    motion2_final = np.copy(motion2_reset)
    motion2_final = rigid_transform(relative, motion2_final)
    motion1_final = motion1_reset

    # joint distance for reset motion1 & motion2
    motion1_final_pos = motion1_final[:, :22 * 3].reshape(-1, 22, 3)
    motion2_final_pos = motion2_final[:, :22 * 3].reshape(-1, 22, 3)
    joint_dist = np.linalg.norm(
        motion1_final_pos[:, :, np.newaxis, :] - motion2_final_pos[:, np.newaxis, :, :],
        axis=-1,
    )
    joint_dist = joint_dist.reshape(-1, 22 * 22)
    return motion1_reset, motion2_reset, motion2_final, joint_dist


class VQMotionDataset(data.Dataset):

    def __init__(self, opt, window_size, normalize=True):
        self.opt = opt
        self.window_size = window_size
        self.normalize = normalize
        self.max_cond_length = opt.max_cond_length
        self.min_cond_length = opt.min_cond_length
        self.max_gt_length = opt.max_gt_length
        self.min_gt_length = window_size  # opt.min_gt_length default 15

        self.max_length = self.max_cond_length + self.max_gt_length - 1
        self.min_length = self.min_cond_length + self.min_gt_length - 1
        self.motion_rep = opt.motion_rep
        self.data_root = pjoin(opt.data_root, 'motions_processed')
        self.normalizer = MotionNormalizer(opt.name)

        ignore_list = []
        try:
            ignore_list = open(os.path.join(self.data_root, "./../split/ignore_list.txt"), "r").readlines()
            ignore_list = [item.strip() for item in ignore_list]
        except Exception as e:
            print(e)

        data_list = []
        try:
            data_list = open(os.path.join(self.data_root, "./../split/train.txt"), "r").readlines()
            data_list = [item.strip() for item in data_list]
        except Exception as e:
            print(e)
        random.shuffle(data_list)

        count_min = 0
        self.lengths = []
        self.motion_list, self.motion_reset_list, self.data_jdist = [], [], []
        for file in tqdm(data_list):
            if file in ignore_list:
                # print("ignore: ", file)
                continue

            motion_name = file.strip()
            file_path_person1 = pjoin(self.data_root, "person1", motion_name + ".npy")
            file_path_person2 = pjoin(self.data_root, "person2", motion_name + ".npy")

            motion1, motion1_swap = load_interhuman_motion(file_path_person1, self.min_length, swap=True)
            motion2, motion2_swap = load_interhuman_motion(file_path_person2, self.min_length, swap=True)

            # `process_motion_np` returns `(seq_len-1, -1)`, so we need to add `1` here.
            if motion1 is None or motion1.shape[0] < self.window_size + 1:
                # print("drop min_length: ", file)
                count_min += 1
                continue

            for mots in [[motion1, motion2], [motion1_swap, motion2_swap]]:
                mot1_reset, mot2_reset, mot2, joint_dist = preprocess_motion(mots)
                self.motion_list.append([mot1_reset.copy(), mot2])
                self.motion_reset_list.append([mot1_reset, mot2_reset])
                self.data_jdist.append(joint_dist)
                self.lengths.append(mot1_reset.shape[0] - self.window_size)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.motion_list), self.cumsum[-1]))
        print(f"drop min_length: {count_min}")
        return

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion1 = self.motion_list[motion_id][0][idx:idx + self.window_size]
        motion2 = self.motion_list[motion_id][1][idx:idx + self.window_size]

        motion1_reset = self.motion_reset_list[motion_id][0][idx:idx + self.window_size]
        motion2_reset = self.motion_reset_list[motion_id][1][idx:idx + self.window_size]
        joint_dist = self.data_jdist[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        if self.normalize:
            motion1 = self.normalizer.forward(motion1)
            motion2 = self.normalizer.forward(motion2)
            motion1_reset = self.normalizer.forward(motion1_reset)
            motion2_reset = self.normalizer.forward(motion2_reset)
            joint_dist = self.normalizer.forward_dist(joint_dist)

        in_motion = np.concatenate([motion1_reset, motion2_reset], axis=1)
        out_motion = np.concatenate([motion1, motion2], axis=1)
        return in_motion, out_motion, joint_dist


def DATALoader(data_cfg, batch_size, window_size, num_workers=8):
    trainSet = VQMotionDataset(data_cfg, window_size)
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
