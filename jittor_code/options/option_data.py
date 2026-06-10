from argparse import Namespace
import os
from os.path import join as pjoin

# ############################### InterHuman
interh_cfg = Namespace()
interh_cfg.name = "InterHuman"
interh_cfg.data_root = "./data/InterHuman/"
interh_cfg.motion_rep = "global"
interh_cfg.nb_joints = 22
interh_cfg.max_motion_length = 300
interh_cfg.max_gt_length = 300
interh_cfg.min_gt_length = 15
interh_cfg.max_cond_length = 1
interh_cfg.min_cond_length = 1
interh_cfg.feet_thre = 0.001
interh_cfg.prev_frames = 0
interh_cfg.dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
interh_cfg.motion_dim = 524
interh_cfg.dist_dim = 484
interh_cfg.start_pos_idx = 0
interh_cfg.end_pos_idx = 66
interh_cfg.dim_joint = 16  # 3+3+6+4=16


inter_clip_cfg = Namespace()
inter_clip_cfg.NAME = "InterCLIP"
inter_clip_cfg.NUM_LAYERS = 8
inter_clip_cfg.NUM_HEADS = 8
inter_clip_cfg.DROPOUT = 0.1
inter_clip_cfg.INPUT_DIM = 258
inter_clip_cfg.LATENT_DIM = 1024
inter_clip_cfg.FF_SIZE = 2048
inter_clip_cfg.ACTIVATION = "gelu"
inter_clip_cfg.MOTION_REP = "global"
inter_clip_cfg.FINETUNE = False
inter_clip_cfg.batch_size = 96
inter_clip_cfg.checkpoint_path = "./checkpoints_eval/interhuman/interclip.ckpt"


# ############################### Inter-X
interx_cfg = Namespace()
interx_cfg.name = "InterX"
interx_cfg.dataset_name = "hhi"
interx_cfg.data_root = "./data/InterX/processed"
interx_cfg.motion_dir = pjoin(interx_cfg.data_root, "motions_norm")
interx_cfg.text_dir = pjoin(interx_cfg.data_root, "texts_processed")
interx_cfg.meta_dir = pjoin(interx_cfg.data_root, "meta")
interx_cfg.nb_joints = 56
interx_cfg.dim_pose = interx_cfg.nb_joints * 12
interx_cfg.max_motion_length = 152
interx_cfg.max_text_len = 35
interx_cfg.motion_dim = 672  # 56 * 6 + 56 * 6
interx_cfg.dist_dim = 484
interx_cfg.start_pos_idx = -1
interx_cfg.end_pos_idx = -1
interx_cfg.dim_joint = 6


interx_eval_cfg = Namespace()
interx_eval_cfg.NAME = "InterXEval"
interx_eval_cfg.dataset_name = "hhi"
interx_eval_cfg.batch_size = 32
interx_eval_cfg.checkpoints_dir = pjoin(os.path.dirname(interx_cfg.data_root), "text2motion", "checkpoints")
interx_eval_cfg.checkpoint_path = None


def configure_interx_paths(data_cfg, eval_cfg=None, data_root=None, eval_model_pth=None):
    """Normalize Inter-X dataset/evaluator paths for train/eval entrypoints."""
    root = data_root or data_cfg.data_root
    root = os.path.normpath(root)
    if os.path.basename(root) != "processed":
        root = pjoin(root, "processed")

    data_cfg.data_root = root
    data_cfg.motion_dir = pjoin(root, "motions_norm")
    data_cfg.text_dir = pjoin(root, "texts_processed")
    data_cfg.meta_dir = pjoin(root, "meta")

    if eval_cfg is not None:
        dataset_root = os.path.dirname(root)
        eval_cfg.checkpoints_dir = pjoin(dataset_root, "text2motion", "checkpoints")
        eval_cfg.checkpoint_path = None
        if eval_model_pth is not None:
            lower = eval_model_pth.lower()
            if lower.endswith((".tar", ".pth", ".pt", ".ckpt", ".pkl")):
                eval_cfg.checkpoint_path = eval_model_pth
            else:
                eval_cfg.checkpoints_dir = eval_model_pth

    return data_cfg, eval_cfg
