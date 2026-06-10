import os
import sys

sys.path.append(sys.path[0] + r"/../")

import h5py
import numpy as np
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin

import torch
from utils.misc import to_torch
import utils.rotation_conversions as geometry
from utils.quaternion import qrot_np, qbetween_np, qmul
from data.human_body_prior.body_model.body_model import BodyModel
from options.option_data import interx_cfg as data_cfg


# https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]


def get_joint_offsets(J, parents):
    offsets = []
    for i in range(len(J)):
        parent_idx = parents[i]
        if parent_idx == -1:
            offsets.append(J[i])  # The root node's offset is its own coordinates
        else:
            offsets.append(J[i] - J[parent_idx])

    offsets = np.array(offsets)
    return offsets


def parse_motion_file(motion_file):
    """
    body_pose (896, 21, 3)
    lhand_pose (896, 15, 3)
    rhand_pose (896, 15, 3)
    jaw_pose (896, 3)
    leye_pose (896, 3)
    reye_pose (896, 3)
    betas (1, 10)
    global_orient (896, 3)
    transl (896, 3)
    gender ()
    """
    data = np.load(motion_file, allow_pickle=True)
    body_pose = data['pose_body'][::downsample]
    left_hand_pose = data['pose_lhand'][::downsample]
    right_hand_pose = data['pose_rhand'][::downsample]
    root_transl = data['trans'][::downsample]
    global_orient = data['root_orient'][::downsample]
    frame_number = body_pose.shape[0]
    jaw_pose = np.zeros((frame_number, 3))
    leye_pose = np.zeros((frame_number, 3))
    reye_pose = np.zeros((frame_number, 3))
    bm = neutral_bm  # gender is ignored for training/evaluation

    pose_seq = []
    with torch.no_grad():
        for fId in range(0, frame_number):
            root_orient = torch.Tensor(global_orient[fId:fId + 1, :]).to(comp_device)
            pose_body = torch.Tensor(body_pose[fId:fId + 1, :].reshape([1, -1])).to(comp_device)
            hand_pose = np.concatenate([left_hand_pose[fId:fId + 1, :], right_hand_pose[fId:fId + 1, :]], axis=1).reshape([1, -1])
            pose_hand = torch.Tensor(hand_pose).to(comp_device)
            trans = torch.Tensor(root_transl[fId:fId + 1]).to(comp_device)
            body = bm(pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient)  # betas is ignored for training/evaluation
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)
    pose_seq_np = pose_seq.detach().cpu().numpy()
    root_transl[:, 1] = root_transl[:, 1] - np.min(pose_seq_np[:, :, 1])  # min-value align to the ground
    final_pose = np.concatenate((global_orient[:, None], body_pose, jaw_pose[:, None], leye_pose[:, None], reye_pose[:, None], left_hand_pose, right_hand_pose, root_transl[:, None]), axis=1)
    return final_pose, pose_seq_np  # [T, 56, 3]


def normalize_transl(motion_p1, motion_p2):
    base = motion_p1[0, -1]
    motion_p2[:, -1] -= base
    motion_p1[:, -1] -= base
    return motion_p1, motion_p2


def get_joint_matrix(motion1_pos, motion2_pos, n_joints=22):
    joint_dist = np.linalg.norm(motion1_pos[:, :, np.newaxis, :] - motion2_pos[:, np.newaxis, :, :], axis=-1)
    joint_dist = joint_dist.reshape(-1, n_joints * n_joints)
    return joint_dist


def forward_kinematics(pose, root_pos, kinematic_tree, skel_joints, do_root_R=True):
    T, J, _ = pose.shape
    pose = to_torch(pose)
    root_pos = to_torch(root_pos)
    pose_mat = geometry.axis_angle_to_matrix(pose)
    offsets = torch.from_numpy(skel_joints).to(pose_mat.device)

    if len(offsets.shape) == 2:
        offsets = offsets.expand(T, -1, -1)
    joints = torch.zeros((T, J, 3)).to(pose_mat.device)
    joints[..., 0, :] = root_pos
    for chain in kinematic_tree:
        if do_root_R:
            matR = pose_mat[:, 0]
        else:
            matR = torch.eye(3).expand((len(T), -1, -1)).detach().to(pose_mat.device)
        for i in range(1, len(chain)):
            matR = torch.matmul(matR, pose_mat[:, chain[i]])
            offset_vec = offsets[:, chain[i]].unsqueeze(-1)
            joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
    return joints


def forward_kinematics_smplx(data):
    bm = neutral_bm  # gender is ignored for training/evaluation
    frame_number = data.shape[0]
    rotations, root_transl = data[:, :-1, :], data[:, -1, :]

    pose_seq = []
    with torch.no_grad():
        for fId in range(0, frame_number):
            root_orient = torch.Tensor(rotations[fId:fId + 1, 0, :]).to(comp_device)
            pose_body = torch.Tensor(rotations[fId:fId + 1, 1:22, :].reshape([1, -1])).to(comp_device)
            pose_hand = torch.Tensor(rotations[fId:fId + 1, 25:, :].reshape([1, -1])).to(comp_device)
            trans = torch.Tensor(root_transl[fId:fId + 1]).to(comp_device)

            body = bm(pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient)  # betas is ignored for training/evaluation
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)
    pose_seq_np = pose_seq.detach().cpu().numpy()
    return pose_seq_np


def normalize_transl_jpos(jpos_p1, jpos_p2, root_transl_p1, root_transl_p2):
    jpos_p1 = jpos_p1 - jpos_p1[0, 0, :]  # root rotation to identity
    jpos_p1 = jpos_p1 + root_transl_p1  # root translation to identity

    jpos_p2 = jpos_p2 - jpos_p2[0, 0, :]  # root rotation to identity
    jpos_p2 = jpos_p2 + root_transl_p2  # root translation to identity
    return jpos_p1, jpos_p2


def normalize_motion_pose(motion, gpos_prev_frame, face_joint_indx=[2, 1, 17, 16]):
    """    
    Standardize motion data so the motion is placed on the ground, the root node’s XZ position is zeroed, and the skeleton’s initial facing direction is +Z.

    Parameters:
    - motion: np.array of shape (seq_len, n_joints+1, 3)
    - n_joints: number of joints
    - prev_frame_index: index of the starting frame used for XZ translation and orientation calculation
    - face_joint_indx: (r_hip, l_hip, sdr_r, sdr_l) hip joint indices used for orientation calculation

    Returns:
    - processed_motion: adjusted position and rotation data, in the same format as the input motion
    """

    # # Extract position and rotation components
    rotations = motion[:, :-1, :]
    root_positions = motion[:, -1, :]

    # XZ at origin
    root_pos_init = gpos_prev_frame
    root_xz = root_pos_init[0] * np.array([1, 0, 1])  # Root joint XZ coordinates (assuming joint 0 is the root)
    root_positions = root_positions - root_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([0, 1, 0]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([0, 0, 1])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.tile(root_quat_init, (root_positions.shape[0], 1))
    root_positions = qrot_np(root_quat_init_for_all, root_positions)

    quant_rotations = geometry.axis_angle_to_quaternion(to_torch(rotations))
    quant_rotations[:, 0] = qmul(to_torch(root_quat_init_for_all).double(), quant_rotations[:, 0])
    rotations = geometry.quaternion_to_axis_angle(quant_rotations).cpu().numpy()

    processed_motion = np.concatenate([rotations, root_positions[:, None, :]], axis=1)
    return processed_motion


def preprosee_motion(name):
    P1_data, P1_pose = parse_motion_file(os.path.join(source_motion_dir, name, 'P1.npz'))
    P2_data, P2_pose = parse_motion_file(os.path.join(source_motion_dir, name, 'P2.npz'))
    P1_data, P2_data = normalize_transl(P1_data, P2_data)
    data = np.concatenate([P1_data, P2_data], axis=-1)  # (T, 56, 6)

    P1_pose = forward_kinematics_smplx(P1_data)
    P2_pose = forward_kinematics_smplx(P2_data)

    P1_data_norm = normalize_motion_pose(P1_data, gpos_prev_frame=P1_pose[0])
    P2_data_norm = normalize_motion_pose(P2_data, gpos_prev_frame=P2_pose[0])
    data_norm = np.concatenate([P1_data_norm, P2_data_norm], axis=-1)  # (T, 56, 6)

    # ! Calculate the distance matrix using the first 22 joints
    P1_pose, P2_pose = P1_pose[:, :22, :], P2_pose[:, :22, :]
    joint_dist = get_joint_matrix(P1_pose, P2_pose)
    return data, data_norm, joint_dist


def prepare_inter_x(data_cfg, mode='train'):
    '''
    1. Read the existing train/val/test motions;
    2. Regenerate motions and global positions from the SMPL-X model;
    '''
    save_motion_dir = pjoin(data_cfg.data_root, 'motions_norm', mode)
    os.makedirs(save_motion_dir, exist_ok=True)

    if mode == 'train':
        split_file = pjoin(data_cfg.data_root, './../splits/train.txt')
        motion_file = pjoin(data_cfg.motion_dir, 'train.h5')
    elif mode == 'val':
        split_file = pjoin(data_cfg.data_root, './../splits/val.txt')
        motion_file = pjoin(data_cfg.motion_dir, 'val.h5')
    elif mode == 'test':
        split_file = pjoin(data_cfg.data_root, './../splits/test.txt')
        motion_file = pjoin(data_cfg.motion_dir, 'test.h5')
    else:
        raise ValueError('Unknown mode')

    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())

    with h5py.File(motion_file, 'r') as mf:
        for name in tqdm(id_list):
            try:
                svae_file_path = pjoin(save_motion_dir, f'{name}.npz')
                if os.path.exists(svae_file_path):
                    continue
                # motion = mf[name][:].astype('float32')
                motion, motion_norm, joint_dist = preprosee_motion(name)

                data = {
                    'motion': motion,
                    'motion_norm': motion_norm,
                    'joint_dist': joint_dist,
                }
                np.savez(svae_file_path, data=data)
            except Exception as e:
                print(e)
                pass

    return


if __name__ == '__main__':
    data_cfg.motion_dir = './data/InterX/processed/motions/'
    source_motion_dir = './data/InterX/motions/'
    neutral_bm_path = './data/body_model/smplx/SMPLX_NEUTRAL.npz'

    # use same body model for all characters in the Inter-X dataset.
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_betas = 10
    downsample = 4
    neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas).to(comp_device)
    bm_parents = neutral_bm.kintree_table[0].long().cpu().numpy()
    bm_tpose = neutral_bm().Jtr[0].cpu().numpy()
    bm_offsets = get_joint_offsets(bm_tpose, bm_parents)
    bm_kinematic_chain = [
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
        [0, 3, 6, 9, 12, 15, 22],
        [15, 23],
        [15, 24],
        [9, 13, 16, 18, 20, 25, 26, 27],
        [20, 28, 29, 30],
        [20, 31, 32, 33],
        [20, 34, 35, 36],
        [20, 37, 38, 39],
        [9, 14, 17, 19, 21, 40, 41, 42],
        [21, 43, 44, 45],
        [21, 46, 47, 48],
        [21, 49, 50, 51],
        [21, 52, 53, 54],
    ]

    prepare_inter_x(data_cfg, mode='train')
    prepare_inter_x(data_cfg, mode='val')
    prepare_inter_x(data_cfg, mode='test')
