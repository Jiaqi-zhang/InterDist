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


'''For use of training text motion matching model, and evaluations'''


class Text2MotionDatasetV2HHI(data.Dataset):

    def __init__(self, opt, is_test, w_vectorizer, unit_length):
        self.opt = opt
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 30 if self.opt.dataset_name == 'hhi_text' else 24
        self.normalizer = MotionNormalizer(opt.name)

        if is_test:
            split_file = pjoin(opt.data_root, './../splits/test.txt')
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
            if is_test:
                file_path = pjoin(opt.motion_dir, 'test', name + '.npz')
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
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
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

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))], axis=0)
            motion_reset = np.concatenate([motion_reset, np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))], axis=0)
            joint_dist = np.concatenate([joint_dist, np.zeros((self.max_motion_length - m_length, joint_dist.shape[1]))], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            motion_reset = motion_reset[:self.max_motion_length]
            joint_dist = joint_dist[:self.max_motion_length]
            m_length = self.max_motion_length

        motion = self.to_rot_6d(motion).float()
        motion_reset = self.to_rot_6d(motion_reset).float()
        return word_embeddings, pos_one_hots, caption, sent_len, motion, motion_reset, joint_dist, m_length, '_'.join(tokens)

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
        ret = torch.reshape(ret, (T, -1))
        return ret


def DATALoader(opt, is_test, batch_size, w_vectorizer, num_workers=8, unit_length=4, shuffle=True):
    dataset = Text2MotionDatasetV2HHI(opt, is_test, w_vectorizer, unit_length)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
