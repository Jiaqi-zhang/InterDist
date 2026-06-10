from utils.jittor_compat import jt
import numpy as np
import os
from argparse import Namespace
from os.path import join as pjoin

from models.evaluator.interx.t2m_eval_modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from utils.word_vectorizer import POS_enumerator


def _load_checkpoint(path, device):
    try:
        return jt.load(path, map_location=device)
    except Exception as jt_error:
        try:
            from tools.convert_torch_checkpoint import load_torch_checkpoint
            return load_torch_checkpoint(path)
        except Exception as torch_zip_error:
            raise RuntimeError(
                f"Failed to load InterX evaluator checkpoint {path}. "
                f"Jittor load error: {jt_error}; torch-zip fallback error: {torch_zip_error}"
            ) from torch_zip_error


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word, pos_size=opt.dim_pos_ohot, hidden_size=opt.dim_text_hidden, output_size=opt.dim_coemb_hidden, device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent, hidden_size=opt.dim_motion_hidden, output_size=opt.dim_coemb_hidden, device=opt.device)

    checkpoint_path = getattr(opt, "checkpoint_path", None)
    if checkpoint_path is None:
        candidates = [
            pjoin(opt.checkpoints_dir, 'text_mot_match', 'model', 'finest.tar'),
            pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
        ]
        checkpoint_path = next((path for path in candidates if os.path.exists(path)), candidates[0])
    checkpoint = _load_checkpoint(checkpoint_path, opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper from %s (Epoch %d) Completed!!' % (checkpoint_path, checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, device, checkpoints_dir, checkpoint_path=None):
        opt = Namespace()
        opt.dim_pose = 56 * 12
        opt.unit_length = 4
        opt.device = device
        opt.checkpoints_dir = checkpoints_dir
        opt.checkpoint_path = checkpoint_path

        opt.dim_movement_dec_hidden = 512
        opt.dim_movement_enc_hidden = 512
        opt.dim_movement_latent = 512

        opt.dataset_name = 'hhi'
        opt.dim_word = 300
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        
        opt.max_motion_length = 150
        opt.max_text_len = 35

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with jt.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.numpy().tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with jt.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.numpy().tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
