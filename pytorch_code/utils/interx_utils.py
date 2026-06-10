import numpy as np
import torch
from data.body_model.body_model import BodyModel
from data.body_model.lbs import batch_rodrigues
import data.rotation_conversions as geometry


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


def get_children_list(parents):
    """  
    Get list of children for each joint  
    """
    children = [[] for _ in range(len(parents))]
    for i, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(i)
    return children


class InterxKinematics():

    def __init__(self):
        self.bm = BodyModel(bm_fname="data/body_model/smplx/SMPLX_NEUTRAL.npz", num_betas=10)
        self.bm.eval()

        self.parents = self.bm.kintree_table[0].long().cpu().numpy()
        
        # the base or resting orientation of each joint in the skeleton. 
        orients = self.bm().full_pose[0].cpu().numpy()  # (1, 165)
        self.orients = orients.reshape(55, 3)  # sum() == 0        
        
        bm_tpose = self.bm().Jtr[0].cpu().numpy()
        self.offsets = get_joint_offsets(bm_tpose, self.parents)
        self.children = get_children_list(self.parents)

        self.bm_kinematic_chain = [
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

        self.names = [
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

    def rot6d_to_axisangle(self, motionrot6d):
        # Root Translation and Velocity
        root = motionrot6d[:, :, -1, :]
        root_trans = root[:, :, :3]
        root_vel = root[:, :, 3:]

        # Whole body pose
        pose = motionrot6d[:, :, :-1, :]
        pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))

        # Root Orientation
        root_orient = pose[:, :, 0, :]
        body_pose = pose[:, :, 1:, :]

        return root_trans, root_vel, root_orient, body_pose
    
    def get_full_pose_quat(self, motionrot6d):
        # Whole body pose
        pose = motionrot6d[:, :, :-1, :]
        # quat: (w, x, y, z)
        pose = geometry.matrix_to_quaternion(geometry.rotation_6d_to_matrix(pose))
        return pose
    
    def get_full_pose_axisAng(self, motionrot6d):
        # Whole body pose
        pose = motionrot6d[:, :, :-1, :]
        # quat: (w, x, y, z)
        pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))
        return pose
    
    def forward_axis_angle(self, motions):
        B, T, J, dim = motions.shape
        self.bm.to(motions.device)
        
        # Root Translation and Velocity
        root_trans = motions[:, :, -1, :]
        body_pose = motions[:, :, :-1, :]
        
        # (B, T, 1+54, 3)
        motions_pos = self.forward_kinematics(root_trans, body_pose)
        return motions_pos

    def forward(self, motions, mode='fk'):
        """
        Args:
            motions: torch.Tensor of shape (B, T, 56, dim) 
            T=64 for vqvae
            dim=6 for 6d rots
        """
        B, T, J, dim = motions.shape

        self.bm.to(motions.device)

        root_trans, root_vel, root_orient, body_pose = self.rot6d_to_axisangle(motions)

        if mode == 'fk':
            # print(root_orient.shape, body_pose.shape)
            # torch.Size([256, 32, 3]) torch.Size([256, 32, 54, 3])
            body_pose = torch.cat(
                [root_orient[:, :, None, :], body_pose], dim=2)  # (B, T, 1+54, 3)
            motions_pos = self.forward_kinematics(root_trans, body_pose)
        elif mode == 'bm':
            motions_pos = []
            # torch.zeros(B, T, J-1, 3, requires_grad=True).to(motions.device)
            for b in range(B):
                bm_output = self.bm(root_orient=root_orient[b], pose_body=body_pose[b, :, 0:21, :].reshape(
                    T, -1), pose_hand=body_pose[b, :, 24:, :].reshape(T, -1))
                motions_pos.append(bm_output.Jtr + root_trans[b].unsqueeze(-2))
                # motions_pos[b] = bm_output.Jtr + root_trans[b].unsqueeze(-2)
            motions_pos = torch.stack(motions_pos, dim=0)
        else:
            raise ValueError('mode must be either fk or bm')

        return motions_pos

    def forward_kinematics(self, root_pos, pose):
        '''  
            root_pos: torch.Tensor of shape (B, T, 3)  
            pose: torch.Tensor of shape (B, T, 55, 3)  
            offsets: torch.Tensor of shape (55, 3)  

            return:
            global_positions: torch.Tensor of shape (B, T, 55, 3)
        '''
        b, t, j, _ = pose.shape
        # Reshape for batch processing
        joints_rotations = batch_rodrigues(
            pose.reshape(-1, 3), dtype=pose.dtype).reshape(b, t, j, 3, 3)

        # Convert offsets to tensor and expand to match batch and time dimensions
        offsets = torch.from_numpy(self.offsets).to(joints_rotations.device)
        offsets = offsets.expand(b, t, -1, -1)

        # Initialize lists to store positions and rotations for each joint
        global_positions_list = []
        global_rotations_list = []

        # Set root positions and rotations
        global_positions_list.append(root_pos)
        global_rotations_list.append(joints_rotations[:, :, 0])

        # Compute global transforms for each joint across all batches and time steps
        for i in range(1, j):
            parent = self.parents[i]
            # Access the parent data from our lists
            parent_rot = global_rotations_list[parent]
            parent_pos = global_positions_list[parent]

            # Calculate rotation and position
            global_rot_i = torch.matmul(parent_rot, joints_rotations[:, :, i])
            global_offset = torch.matmul(
                parent_rot, offsets[:, :, i].unsqueeze(-1)).squeeze(-1)
            global_pos_i = parent_pos + global_offset

            # Store results in lists
            global_rotations_list.append(global_rot_i)
            global_positions_list.append(global_pos_i)

        # Stack all positions along the joint dimension
        global_positions = torch.stack(global_positions_list, dim=2)

        return global_positions
