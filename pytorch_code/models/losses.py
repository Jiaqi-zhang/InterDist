import torch
from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
from data.quaternion import cont6d_to_matrix, qbetween
from utils.utils import MotionNormalizerTorch, fid_l, fid_r, face_joint_indx
from utils.interx_utils import InterxKinematics


class Geometric_Losses:

    def __init__(self, norm_data_root, recons_loss, joints_num, dataset_name, device, ex_loss=False):

        if recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss()
        elif recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss()

        self.joints_num = joints_num
        self.fids = [*fid_l, *fid_r]
        self.dataset_name = dataset_name
        self.normalizer = MotionNormalizerTorch(dataset_name, norm_data_root, device)
        if self.dataset_name == 'InterX' and ex_loss:
            self.kinematics = InterxKinematics()

    def calc_foot_contact(self, motion, pred_motion):
        if self.dataset_name == 'InterHuman':
            B, T, _ = motion.shape
            motion = motion[..., :self.joints_num * 3]
            motion = motion.reshape(B, T, self.joints_num, 3)

            pred_motion = pred_motion[..., :self.joints_num * 3]
            pred_motion = pred_motion.reshape(B, T, self.joints_num, 3)

        feet_vel = motion[:, 1:, self.fids, :] - motion[:, :-1, self.fids, :]
        pred_feet_vel = pred_motion[:, 1:, self.fids, :] - pred_motion[:, :-1, self.fids, :]
        feet_h = motion[:, :-1, self.fids, 1]
        pred_feet_h = pred_motion[:, :-1, self.fids, 1]
        # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

        ## Calculate contacts
        thres = 0.001
        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor([0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

        feet_x = (feet_vel[..., 0])**2
        feet_y = (feet_vel[..., 1])**2
        feet_z = (feet_vel[..., 2])**2
        contact = ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)

        fc_loss = self.l1_criterion(pred_feet_vel[contact], torch.zeros_like(pred_feet_vel)[contact])
        if torch.isnan(fc_loss):
            fc_loss = torch.tensor(0).to(motion.device)
            if contact.sum() != 0:
                print("FC nan but contact not 0")
        return fc_loss

    def calc_bone_lengths(self, motion):
        if self.dataset_name == 'InterHuman':
            motion_pos = motion[..., :self.joints_num * 3]
            motion_pos = motion_pos.reshape(motion_pos.shape[0], motion_pos.shape[1], self.joints_num, 3)
        elif self.dataset_name == 'InterX':
            motion_pos = motion
        bones = []
        for chain in kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                bone = (motion_pos[..., chain[i], :] - motion_pos[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B,T,P,1]
                bones.append(bone)

        return torch.cat(bones, dim=-1)

    def calc_loss_geo(self, pred_rot, gt_rot, eps=1e-7):
        if self.dataset_name == "InterHuman":
            pred_rot = pred_rot.reshape(pred_rot.shape[0], pred_rot.shape[1], -1, 6)
            gt_rot = gt_rot.reshape(gt_rot.shape[0], gt_rot.shape[1], -1, 6)

        pred_m = cont6d_to_matrix(pred_rot).reshape(-1, 3, 3)
        gt_m = cont6d_to_matrix(gt_rot).reshape(-1, 3, 3)

        m = torch.bmm(gt_m, pred_m.transpose(1, 2))  #batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))

        return torch.mean(theta)
    
    def calc_loss_dist_interH(self, pred_mot, gt_mot):
        def cal_dist(mot):
            mot1, mot2 = mot
            mot1 = self.normalizer.backward(mot1)
            mot2 = self.normalizer.backward(mot2)
            
            bs, T, _ = mot1.shape
            mot1_pos = mot1[..., :22*3].reshape(bs, T, 22, 3).reshape(-1, 22, 3)
            mot2_pos = mot2[..., :22*3].reshape(bs, T, 22, 3).reshape(-1, 22, 3)            
            joint_dist = torch.norm(  
                mot1_pos[:, :, None, :] - mot2_pos[:, None, :, :], dim=-1
            )
            joint_dist = joint_dist.reshape(-1, 22*22)
            return joint_dist
        
        pred_dist = cal_dist(pred_mot)
        gt_dist = cal_dist(gt_mot)
        dist_loss = self.l1_criterion(pred_dist, gt_dist)
        return dist_loss
    
    def calc_loss_dist_interX(self, pred_pos, gt_pos):
        # pred_pos: (bs, T, 55, 3)
        def cal_dist(pos):
            mot1_pos, mot2_pos = pos[0][:, :, :22, :], pos[1][:, :, :22, :]
            mot1_pos = mot1_pos.reshape(-1, 22, 3)
            mot2_pos = mot2_pos.reshape(-1, 22, 3)
                
            joint_dist = torch.norm(  
                mot1_pos[:, :, None, :] - mot2_pos[:, None, :, :], dim=-1
            )
            joint_dist = joint_dist.reshape(-1, 22*22)
            return joint_dist
        
        pred_dist = cal_dist(pred_pos)
        gt_dist = cal_dist(gt_pos)
        dist_loss = self.l1_criterion(pred_dist, gt_dist)
        return dist_loss
    
    def forward(self, motions, pred_motion, only_rec=False):
        if only_rec:
            loss_rec = self.l1_criterion(pred_motion, motions)
            return loss_rec
            
        if self.dataset_name == 'InterHuman':            
            loss_rec = self.l1_criterion(pred_motion[..., :-4], motions[..., :-4])

            loss_explicit = self.l1_criterion(pred_motion[:, :, :self.joints_num * 3], motions[:, :, :self.joints_num * 3])

            loss_vel = self.l1_criterion(pred_motion[:, 1:, :self.joints_num * 3] - pred_motion[:, :-1, :self.joints_num * 3],
                                         motions[:, 1:, :self.joints_num * 3] - motions[:, :-1, :self.joints_num * 3])

            loss_bn = self.l1_criterion(self.calc_bone_lengths(pred_motion), self.calc_bone_lengths(motions))

            loss_geo = self.calc_loss_geo(pred_motion[..., self.joints_num * 6:self.joints_num * 6 + (self.joints_num - 1) * 6],
                                          motions[..., self.joints_num * 6:self.joints_num * 6 + (self.joints_num - 1) * 6])

            loss_fc = self.calc_foot_contact(self.normalizer.backward(motions), self.normalizer.backward(pred_motion))

            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, None, None
            
        elif self.dataset_name == 'InterX':            
            pred_motion = pred_motion.reshape(pred_motion.shape[0], pred_motion.shape[1], 56, 6)
            motions = motions.reshape(motions.shape[0], motions.shape[1], 56, 6)                
            loss_rec = self.l1_criterion(pred_motion, motions)
            
            # loss_root_vel = self.l1_criterion(pred_motion[:, :, -1, :3], motions[:, :, -1, :3])
            
            B, T, J, D = motions.shape
            pred_motion = self.normalizer.backward(pred_motion.reshape(B, T, -1)).reshape(B, T, J, D)
            motions = self.normalizer.backward(motions.reshape(B, T, -1)).reshape(B, T, J, D)
            
            loss_geo = self.calc_loss_geo(pred_motion[:, :, :-1, :], motions[:, :, :-1, :])
            pred_motions_pos = self.kinematics.forward(pred_motion)
            motions_pos = self.kinematics.forward(motions)

            loss_explicit = self.l1_criterion(pred_motions_pos, motions_pos)
            loss_vel = self.l1_criterion(pred_motions_pos[:, 1:, :, :] - pred_motions_pos[:, :-1, :, :], motions_pos[:, 1:, :, :] - motions_pos[:, :-1, :, :])
            
            loss_bn = torch.zeros_like(loss_rec)            
            loss_fc = self.calc_foot_contact(motions_pos, pred_motions_pos)
            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, motions_pos, pred_motions_pos
