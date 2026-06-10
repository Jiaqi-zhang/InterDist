from argparse import Namespace

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


# ############################### Inter-X
interx_cfg = Namespace()
interx_cfg.name = "InterX"
interx_cfg.dataset_name = "hhi"
interx_cfg.data_root = "./data/InterX/processed/"
interx_cfg.motion_dir = interx_cfg.data_root + "motions_norm/"
interx_cfg.text_dir = interx_cfg.data_root + "texts_processed/"
interx_cfg.meta_dir = interx_cfg.data_root + "meta/"
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
interx_eval_cfg.checkpoints_dir = interx_cfg.data_root + "./../text2motion/checkpoints/"
