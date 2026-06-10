import numpy as np
from os.path import join as pjoin

from utils.jittor_compat import jt
from models.evaluator.interhuman.t2m_eval_modules import InterCLIP


def build_models(cfg, data_root):
    model = InterCLIP(cfg)

    checkpoint_path = getattr(cfg, "checkpoint_path", None) or pjoin(data_root, 'eval_model/interclip.ckpt')
    checkpoint = jt.load(checkpoint_path, map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, data_root, device):

        self.model = build_models(cfg, data_root)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with jt.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = jt.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.numpy().tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            motion_lens_list = [int(x) for x in motion_lens.numpy().tolist()]
            cur_len = jt.LongTensor([min(T, m_len) for m_len in motion_lens_list]).to(self.device)
            padded_len = int(cur_len.max().item())

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']
            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with jt.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = jt.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.numpy().tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            motion_lens_list = [int(x) for x in motion_lens.numpy().tolist()]
            cur_len = jt.LongTensor([min(T, m_len) for m_len in motion_lens_list]).to(self.device)
            padded_len = int(cur_len.max().item())

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
