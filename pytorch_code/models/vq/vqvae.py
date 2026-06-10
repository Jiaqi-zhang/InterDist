import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ


class VQVAE(nn.Module):

    def __init__(self, args, input_dim=12, nb_code=1024, code_dim=512, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=2, 
                 dilation_growth_rate=3, activation='relu', norm=None, n_heads=8, ff_size=1024):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.motion_dim = args.motion_dim
        self.joints_num = args.nb_joints
        self.dataname = args.dataname

        if self.dataname == "InterHuman":
            input_dist_dim = self.joints_num
            filter_s = None
            stride_s = None
            spatial_upsample = (2.2, 2)
            # gcn = True
            gcn = False

        elif self.dataname == "InterX":
            input_dist_dim = 22
            filter_s = 6
            stride_s = 3
            spatial_upsample = (3.5, 3.3)
            gcn = False

        self.encoder_dist = Encoder(
            input_dist_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, 
            filter_s=None, stride_s=None, gcn=False,
        )
        
        self.encoder_mot = Encoder(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, filter_s=filter_s, stride_s=stride_s, gcn=gcn)
        self.decoder = Decoder(
            input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, 
            spatial_upsample=spatial_upsample, gcn=gcn, n_heads=n_heads, ff_size=ff_size,
        )

        assert args.num_quantizers == 1
        rvqvae_config = {
            'args': args,
            'nb_code': nb_code,
            'code_dim': code_dim,
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        self.quantizer_dist = ResidualVQ(**rvqvae_config)
    
    def preprocess(self, x, joint_dist):
        if self.dataname == "InterHuman":
            '''
            x: B, T, 22*3+22*3+21*6+4=262
            '''
            bs, seq, _ = x.shape            
            pos = x[..., :self.joints_num * 3].reshape([bs, seq, self.joints_num, 3])
            vel = x[..., self.joints_num * 3:self.joints_num * 6].reshape([bs, seq, self.joints_num, 3])
            rot = x[..., self.joints_num * 6:self.joints_num * 6 + (self.joints_num - 1) * 6].reshape([bs, seq, self.joints_num - 1, 6])
            rot = torch.cat([torch.zeros(bs, seq, 1, 6).to(x.device), rot], dim=2)
            feet = x[:, :, -4:].reshape([bs, seq, 1, 4]).repeat(1, 1, self.joints_num, 1)
            joints = torch.cat([pos, vel, rot, feet], dim=-1)  # B, T, 22, 3+3+6+4=16            
            joints = joints.permute(0, 3, 2, 1).float()  # B, D=12, J=22, T
            
            joint_dist = joint_dist.reshape([joint_dist.shape[0], joint_dist.shape[1], self.joints_num, -1])
            joint_dist = joint_dist.permute(0, 3, 2, 1).float()  # B, D=22, J=22, T
        else:
            # joints: (bs*2, t, d)
            joints = x.reshape(x.shape[0], x.shape[1], self.joints_num, -1)  # (bs*2, T, 56, 6)            
            joints = joints.permute(0, 3, 2, 1).float()
            
            joint_dist = joint_dist.reshape([joint_dist.shape[0], joint_dist.shape[1], 22, 22])
            joint_dist = joint_dist.permute(0, 3, 2, 1).float()  # (B, D, 56, T)
        
        return joints, joint_dist

    def postprocess(self, mot1, mot2):
        mot1 = mot1.permute(0, 3, 2, 1).float()  # (B, T, J, D)
        mot2 = mot2.permute(0, 3, 2, 1).float()  # (B, T, J, D)

        def reformat(x):
            pos = x[:, :, :, :3].reshape([bs, seq, -1])
            vel = x[:, :, :, 3:6].reshape([bs, seq, -1])
            rot = x[:, :, 1:, 6:6 + 6].reshape([bs, seq, -1])
            feet = x[:, :, 0, -4:].reshape([bs, seq, -1])
            x = torch.cat([pos, vel, rot, feet], dim=-1)
            return x
        
        bs, seq, _, _ = mot1.shape
        if self.dataname == "InterHuman":
            mot1 = reformat(mot1)
            mot2 = reformat(mot2)
        else:
            mot1 = mot1.reshape([bs, seq, -1])
            mot2 = mot2.reshape([bs, seq, -1])
        return mot1, mot2

    def forward(self, x, joint_dist, verbose=False):
        bs, num_seq, dim = x.shape
        assert dim == self.motion_dim
        mot1, mot2 = x[..., :self.motion_dim//2], x[..., self.motion_dim//2:]
        mot = torch.cat([mot1, mot2], dim=0)  # (2*bs, T, d//2)

        # Encode
        mot, joint_dist = self.preprocess(mot, joint_dist)  # (B, D=12, J=22, T)
        mot = self.encoder_mot(mot)  # (B, D, 5, T/4)
        joint_dist = self.encoder_dist(joint_dist)  # (B, D, 5, T/4)

        # quantization
        mot_shape = mot.shape
        dist_shape = joint_dist.shape
        mot = mot.reshape(mot.shape[0], mot.shape[1], -1)  # (B, D, J*T/4)
        joint_dist = joint_dist.reshape(joint_dist.shape[0], joint_dist.shape[1], -1)

        m_quantized, m_code_idx, m_commit_loss, m_perplexity = self.quantizer(mot, sample_codebook_temp=0.5)
        d_quantized, d_code_idx, d_commit_loss, d_perplexity = self.quantizer_dist(joint_dist, sample_codebook_temp=0.5)        
        commit_loss = (m_commit_loss + d_commit_loss) / 2
        perplexity = (m_perplexity + d_perplexity) / 2
        
        m_quantized = m_quantized.reshape(mot_shape)  # (B, D, 5, T/4)
        d_quantized = d_quantized.reshape(dist_shape)  # (B, D, 5, T/4)

        ## decoder
        mot1_quantized, mot2_quantized = m_quantized[:bs], m_quantized[bs:]
        mot1, mot2 = self.decoder(mot1_quantized, mot2_quantized, d_quantized, input_size=num_seq)
        mot1, mot2 = self.postprocess(mot1, mot2)        
        return mot1, mot2, commit_loss, perplexity

    def encode(self, x, joint_dist):
        bs, num_seq, dim = x.shape
        assert dim == self.motion_dim
        mot1, mot2 = x[..., :self.motion_dim//2], x[..., self.motion_dim//2:]
        mot = torch.cat([mot1, mot2], dim=0)  # (2*bs, T, d//2)
        
        # Encode
        mot, joint_dist = self.preprocess(mot, joint_dist)  # (B, D=12, J=22, T)
        mot = self.encoder_mot(mot)  # (B, D, 5, T/4)
        joint_dist = self.encoder_dist(joint_dist)  # (B, D, 5, T/4)
        
        # quantization
        mot_shape = mot.shape
        dist_shape = joint_dist.shape
        mot = mot.reshape(mot.shape[0], mot.shape[1], -1)  # (B, D, J*T/4)
        joint_dist = joint_dist.reshape(joint_dist.shape[0], joint_dist.shape[1], -1)

        m_code_idx, m_all_codes = self.quantizer.quantize(mot, return_latent=True)
        d_code_idx, d_all_codes = self.quantizer_dist.quantize(joint_dist, return_latent=True)
        
        # code_idx: (B, J*T, 1), all_codes: (1, B, D, J*T)
        m_code_idx, m_all_codes = m_code_idx[..., 0], m_all_codes[0]
        d_code_idx, d_all_codes = d_code_idx[..., 0], d_all_codes[0]        
        
        m1_code_idx, m2_code_idx = m_code_idx[:bs], m_code_idx[bs:]  # (B, J*T/4)
        m1_all_codes, m2_all_codes = m_all_codes[:bs], m_all_codes[bs:]  # (B, D, J*T/4)
        
        return m1_code_idx, m2_code_idx, d_code_idx, m1_all_codes, m2_all_codes, d_all_codes

    def forward_decoder(self, mot1_cidx, mot2_cidx, dist_cidx, num_seq, soft_lookup=False):
        mot_shape = mot1_cidx.shape

        if not soft_lookup:
            mot1 = self.quantizer.get_codes_from_indices(mot1_cidx[..., None])[0]
            mot2 = self.quantizer.get_codes_from_indices(mot2_cidx[..., None])[0]
            joint_dist = self.quantizer_dist.get_codes_from_indices(dist_cidx[..., None])[0]
            # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        else:
            mot1 = self.quantizer.get_soft_codes_from_probs(mot1_cidx)
            mot2 = self.quantizer.get_soft_codes_from_probs(mot2_cidx)
            joint_dist = self.quantizer_dist.get_soft_codes_from_probs(dist_cidx)

        # decoder
        mot1 = mot1.reshape(mot_shape[0], 5, mot_shape[1]//5, -1).permute(0, 3, 1, 2)
        mot2 = mot2.reshape(mot_shape[0], 5, mot_shape[1]//5, -1).permute(0, 3, 1, 2)
        joint_dist = joint_dist.reshape(mot_shape[0], 5, mot_shape[1]//5, -1).permute(0, 3, 1, 2)
        mot1, mot2 = self.decoder(mot1, mot2, joint_dist, input_size=num_seq)
        mot1, mot2 = self.postprocess(mot1, mot2)
        return mot1, mot2


class HumanVQVAE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.vqvae = VQVAE(
            cfg, cfg.dim_joint, cfg.nb_code, cfg.code_dim, cfg.code_dim, cfg.down_t, 
            cfg.stride_t, cfg.width, cfg.depth, cfg.dilation_growth_rate, 
            activation=cfg.vq_act, norm=cfg.vq_norm,
            n_heads=cfg.n_heads, ff_size=cfg.ff_size,
        )

    def forward(self, x, joint_dist, verbose=False):
        return self.vqvae(x, joint_dist, verbose)
    
    def encode(self, x, joint_dist):
        res = self.vqvae.encode(x, joint_dist)
        mot1_cidx, mot2_cidx, dist_cidx, _, _, _ = res        
        return mot1_cidx, mot2_cidx, dist_cidx
    
    def decode(self, mot1_cidx, mot2_cidx, dist_cidx, mot_length):
        pred_motion1, pred_motion2 = self.vqvae.forward_decoder(mot1_cidx, mot2_cidx, dist_cidx, num_seq=mot_length)
        return pred_motion1, pred_motion2
