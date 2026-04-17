import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin

import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from utils.misc import to_torch
import utils.rotation_conversions as geometry
from utils.utils import MotionNormalizer


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''


class Text2MotionDatasetV2HHI(data.Dataset):

    def __init__(self, opt, w_vectorizer, unit_length=4, shuffle=True, is_training=True):
        self.opt = opt
        self.shuffle = shuffle
        self.w_vectorizer = w_vectorizer
        self.unit_length = unit_length

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 30 if self.opt.dataset_name == 'hhi_text' else 24
        self.normalizer = MotionNormalizer(opt.name)

        if is_training:
            split_file = pjoin(opt.data_root, './../splits/train.txt')
        else:
            split_file = pjoin(opt.data_root, './../splits/val.txt')

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            if is_training:
                file_path = pjoin(opt.motion_dir, 'train', name + '.npz')
            else:
                file_path = pjoin(opt.motion_dir, 'val', name + '.npz')
            data = np.load(file_path, allow_pickle=True)['data'].item()
            motion = data['motion'].astype('float32')
            motion_reset = data['motion_norm'].astype('float32')
            joint_dist = data['joint_dist'].astype('float32')

            if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                continue
            text_data = []
            flag = False
            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        exit(-1)
            if flag:
                data_dict[name] = {
                    'motion': motion,
                    'motion_reset': motion_reset,
                    'joint_dist': joint_dist,
                    'length': len(motion),
                    'text': text_data,
                }
                new_name_list.append(name)
                length_list.append(len(motion))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, motion_reset, joint_dist = data['motion'], data['motion_reset'], data['joint_dist']
        m_length, text_list = data['length'], data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10 and self.shuffle:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion_reset = motion_reset[idx:idx + m_length]
        joint_dist = joint_dist[idx:idx + m_length]
        """Z Normalization"""
        T, J, D = motion.shape  # (T, 56, 6)
        mot_lst = []
        for mot in [motion, motion_reset]:
            mot = self.to_rot_6d(mot)
            mot1, mot2 = mot[..., :6], mot[..., 6:]

            mot1 = mot1.reshape(T, -1)
            mot1 = self.normalizer.forward(mot1)
            mot2 = mot2.reshape(T, -1)
            mot2 = self.normalizer.forward(mot2)

            mot = np.concatenate([mot1, mot2], axis=1)
            mot_lst.append(mot)
        motion, motion_reset = mot_lst
        joint_dist = self.normalizer.forward_dist(joint_dist)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
            motion_reset = np.concatenate([motion_reset, np.zeros((self.max_motion_length - m_length, motion_reset.shape[1]))], axis=0)
            joint_dist = np.concatenate([joint_dist, np.zeros((self.max_motion_length - m_length, joint_dist.shape[1]))], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            motion_reset = motion_reset[:self.max_motion_length]
            joint_dist = joint_dist[:self.max_motion_length]
            m_length = self.max_motion_length

        return caption, motion, motion_reset, joint_dist, m_length

    def to_rot_6d(self, data):
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
        T = ret.shape[0]
        ret = ret.cpu().numpy()
        return ret


def DATALoader(opt, w_vectorizer, batch_size, unit_length=4, num_workers=8, shuffle=True, is_training=True):

    train_loader = torch.utils.data.DataLoader(
        Text2MotionDatasetV2HHI(opt, w_vectorizer, unit_length=unit_length, shuffle=shuffle, is_training=is_training),
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
