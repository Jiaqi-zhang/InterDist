import os
import numpy as np
import codecs as cs

import torch
from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin
from utils.misc import to_torch
import utils.rotation_conversions as geometry
from utils.utils import MotionNormalizer


def get_mean_std(data_lst, save_meta_dir, tag=''):
    mean_file_path = pjoin(save_meta_dir, f'interx_mean{tag}.npy')
    std_file_path = pjoin(save_meta_dir, f'interx_std{tag}.npy')
    if os.path.exists(mean_file_path) and os.path.exists(std_file_path):
        mean = np.load(mean_file_path)
        std = np.load(std_file_path)
        return mean, std

    tmp_data = []
    if len(data_lst[0]) == 2:
        for data in data_lst:
            tmp_data.append(data[0])
            tmp_data.append(data[1])
    else:
        tmp_data = data_lst
    tmp_data = np.concatenate(tmp_data, axis=0)
    mean = np.mean(tmp_data, axis=0)
    std = np.std(tmp_data, axis=0)
    idx = std < 1e-5
    std[idx] = 1
    np.save(pjoin(save_meta_dir, f'mean{tag}.npy'), mean)
    np.save(pjoin(save_meta_dir, f'std{tag}.npy'), std)
    return mean, std


class VQMotionDataset(data.Dataset):

    def __init__(self, opt, window_size):
        self.opt = opt
        self.window_size = window_size
        self.num_person = 2
        split_file = pjoin(opt.data_root, './../splits/train.txt')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.data, self.data_reset, self.data_jdist = [], [], []
        self.lengths = []
        count_min = 0
        for name in tqdm(id_list):
            file_path = pjoin(opt.motion_dir, 'train', name + '.npz')
            data = np.load(file_path, allow_pickle=True)['data'].item()
            if data['motion'].shape[0] < self.window_size:
                count_min += 1
                continue

            motion = self.to_rot_6d(data['motion'].astype('float32'))
            motion_reset = self.to_rot_6d(data['motion_norm'].astype('float32'))
            joint_dist = data['joint_dist'].astype('float32')

            T, J, D = motion.shape  # (T, 56, 12)
            mot1 = motion[..., :6].reshape(T, -1)
            mot2 = motion[..., 6:].reshape(T, -1)
            self.data.append([mot1, mot2])

            mot1_reset = motion_reset[..., :6].reshape(T, -1)
            mot2_reset = motion_reset[..., 6:].reshape(T, -1)
            self.data_reset.append([mot1_reset, mot2_reset])

            self.data_jdist.append(joint_dist)
            self.lengths.append(motion.shape[0] - self.window_size)

        assert len(self.data) == len(self.data_reset) == len(self.data_jdist)
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
        print(f"drop min_length: {count_min}")

        # calculate mean and std
        save_meta_dir = './data/stats/'
        os.makedirs(save_meta_dir, exist_ok=True)
        if not os.path.exists(pjoin(save_meta_dir, 'interx_mean_reset.npy')):
            get_mean_std(self.data_reset, save_meta_dir, tag='_reset')
            get_mean_std(self.data_jdist, save_meta_dir, tag='_dist')

        self.normalizer = MotionNormalizer(opt.name)
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
        motion1 = self.data[motion_id][0][idx:idx + self.window_size]
        motion2 = self.data[motion_id][1][idx:idx + self.window_size]

        motion1_reset = self.data_reset[motion_id][0][idx:idx + self.window_size]
        motion2_reset = self.data_reset[motion_id][1][idx:idx + self.window_size]
        joint_dist = self.data_jdist[motion_id][idx:idx + self.window_size]

        "Z Normalization"
        motion1 = self.normalizer.forward(motion1)
        motion2 = self.normalizer.forward(motion2)
        out_motion = np.concatenate([motion1, motion2], axis=1)

        motion1_reset = self.normalizer.forward(motion1_reset)
        motion2_reset = self.normalizer.forward(motion2_reset)
        in_motion = np.concatenate([motion1_reset, motion2_reset], axis=1)

        joint_dist = self.normalizer.forward_dist(joint_dist)

        return in_motion, out_motion, joint_dist

    def to_rot_6d(self, data):
        '''
        data: (T x 56 x 6), 55+1
        '''
        joints3D = np.expand_dims(data[:, -1, :], axis=1)
        joints3D = to_torch(joints3D)
        ret_tr = joints3D[:, 0, :]

        pose = data[:, :-1, :]

        pose = to_torch(pose)
        pose_all = []
        for ii in range(self.num_person):
            pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:, :, 3 * ii:3 * ii + 3])))
        ret = torch.cat(pose_all, dim=2)
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        for ii in range(self.num_person):
            padded_tr[:, 6 * ii:6 * ii + 3] = ret_tr[:, 3 * ii:3 * ii + 3]
        ret = torch.cat((ret, padded_tr[:, None]), 1)
        
        ret = ret.cpu().numpy()
        return ret


def DATALoader(data_cfg, batch_size, window_size, num_workers=8, shuffle=True):
    trainSet = VQMotionDataset(data_cfg, window_size)
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
