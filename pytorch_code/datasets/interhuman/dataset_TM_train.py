import os
import random
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

import torch
from torch.utils import data
from utils.preprocess import load_interhuman_motion
from utils.quaternion import qmul_np, qinv_np, qrot_np
from torch.utils.data._utils.collate import default_collate
from utils.utils import process_motion_np, rigid_transform, MotionNormalizer


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''


class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, shuffle=True, is_training=True):
        self.opt = opt
        self.shuffle = shuffle
        self.is_training = is_training

        self.max_cond_length = opt.max_cond_length
        self.min_cond_length = opt.min_cond_length
        self.max_gt_length = opt.max_gt_length
        self.min_gt_length = opt.min_gt_length
        self.max_length = self.max_cond_length + self.max_gt_length - 1
        self.min_length = self.min_cond_length + self.min_gt_length - 1
        self.motion_rep = opt.motion_rep
        self.data_root = pjoin(opt.data_root, 'motions_processed')
        self.normalizer = MotionNormalizer(opt.name)

        self.data_list = []
        self.motion_dict = {}

        ignore_list = []
        try:
            ignore_list = open(os.path.join(self.data_root, "./../split/ignore_list.txt"), "r").readlines()
            ignore_list = [item.strip() for item in ignore_list]
        except Exception as e:
            print(e)

        data_list = []
        if is_training:
            try:
                data_list = open(os.path.join(self.data_root, "./../split/train.txt"), "r").readlines()
                data_list = [item.strip() for item in data_list]
            except Exception as e:
                print(e)
        else:
            try:
                data_list = open(os.path.join(self.data_root, "./../split/val.txt"), "r").readlines()
                data_list = [item.strip() for item in data_list]
            except Exception as e:
                print(e)
        random.shuffle(data_list)

        index = 0
        count_min = 0

        mot_len = []
        for file in tqdm(data_list):
            if file in ignore_list:
                # print("ignore: ", file)
                continue

            motion_name = file.strip()
            file_path_person1 = pjoin(self.data_root, "person1", motion_name + ".npy")
            file_path_person2 = pjoin(self.data_root, "person2", motion_name + ".npy")
            text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")

            texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
            texts_swap = [
                item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right").replace("clockwise", "tmp").replace("counterclockwise",
                                                                                                                                                   "clockwise").replace("tmp", "counterclockwise")
                for item in texts
            ]

            motion1, motion1_swap = load_interhuman_motion(file_path_person1, self.min_length, swap=True)
            motion2, motion2_swap = load_interhuman_motion(file_path_person2, self.min_length, swap=True)
            if motion1 is None:
                # print("drop min_length: ", file)
                count_min += 1
                continue

            mot_len.append(len(motion1))

            self.motion_dict[index] = [motion1, motion2]
            self.data_list.append({"name": motion_name, "motion_id": index, "swap": False, "texts": texts})

            if is_training:
                self.motion_dict[index + 1] = [motion1_swap, motion2_swap]
                self.data_list.append({"name": motion_name, "motion_id": index + 1, "swap": True, "texts": texts_swap})
            index += 2
        print("total dataset: ", len(self.data_list))
        print(f"drop min_length: {count_min}")
        return

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len() * 1

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]
        text = random.choice(data["texts"]).strip()
        full_motion1, full_motion2 = self.motion_dict[motion_id]

        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]
        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length)
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        if np.random.rand() > 0.5:
            motion1, motion2 = motion2, motion1

        motion1_reset, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2_reset, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)

        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        
        # rigid_transform returns a reference value, so a deep copy is needed here.
        motion1_final = np.copy(motion1_reset)
        motion2_final = np.copy(motion2_reset)
        motion2_final = rigid_transform(relative, motion2_final)

        # joint distance for reset motion1 & motion2
        motion1_final_pos = motion1_final[:, :22 * 3].reshape(-1, 22, 3)
        motion2_final_pos = motion2_final[:, :22 * 3].reshape(-1, 22, 3)
        joint_dist = np.linalg.norm(
            motion1_final_pos[:, :, np.newaxis, :] - motion2_final_pos[:, np.newaxis, :, :],
            axis=-1,
        )
        joint_dist = joint_dist.reshape(-1, 22 * 22)
        "Z Normalization before padding"
        motion1_final = self.normalizer.forward(motion1_final)
        motion2_final = self.normalizer.forward(motion2_final)
        motion1_reset = self.normalizer.forward(motion1_reset)
        motion2_reset = self.normalizer.forward(motion2_reset)
        joint_dist = self.normalizer.forward_dist(joint_dist)

        gt_length = len(motion1_final)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = motion1_final.shape[1]
            motion1_final = np.concatenate((motion1_final, np.zeros((padding_len, D))), axis=0)
            motion2_final = np.concatenate((motion2_final, np.zeros((padding_len, D))), axis=0)
            motion1_reset = np.concatenate((motion1_reset, np.zeros((padding_len, D))), axis=0)
            motion2_reset = np.concatenate((motion2_reset, np.zeros((padding_len, D))), axis=0)
            joint_dist = np.concatenate((joint_dist, np.zeros((padding_len, 22 * 22))), axis=0)

        assert len(motion1_final) == self.max_gt_length
        assert len(motion2_final) == self.max_gt_length
        assert len(motion2_reset) == self.max_gt_length

        gt_motion_norm = np.concatenate((motion1_final, motion2_final), axis=-1)
        gt_set_motion_norm = np.concatenate((motion1_final, motion2_reset), axis=-1)

        return text, gt_motion_norm, gt_set_motion_norm, joint_dist, gt_length


def DATALoader(opt, batch_size, num_workers=8, shuffle=True, is_training=True):

    train_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(opt, shuffle=shuffle, is_training=is_training),
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
