import numpy as np
from tqdm import tqdm
from argparse import Namespace
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

import torch
import options.option_vq as option_vq
from models.vq.vqvae import HumanVQVAE
from utils.utils import fixseed
from utils import eval_t2hh as eval_t2hh
from utils.utils import is_float, is_number


def load_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------', '------------ Options -------------', '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt.is_train = False
    opt.device = device
    return opt


def load_vq_model(opt, model_path):
    # if not use_opt_file:
    #     opt = args
    # else:
    #     opt = load_opt(pjoin(model_dir, 'opt.txt'), args.device)
        
    opt.motion_dim = data_cfg.motion_dim
    opt.dist_dim = data_cfg.dist_dim
    opt.dim_joint = data_cfg.dim_joint
    opt.nb_joints = data_cfg.nb_joints

    net = HumanVQVAE(opt)
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    nb_iter = ckpt['nb_iter'] if 'nb_iter' in ckpt else -1
    print(f'Loading checkpoint from {model_path} completed!, Nb_iter: [{nb_iter}]')
    return net, nb_iter


##### ---- Exp dirs ---- #####
# ======> for InterHuman
# mm_num_samples = 100  # 100 samples for MM
# mm_num_repeats = 30
# mm_num_times = 10
# diversity_times = 300
# batch_size = 96

# ======> for InterX
# mm_num_samples = 100
# mm_num_repeats = 30
# mm_num_times = 10
# diversity_times = 300
# replication_times = 20
# batch_size = 32

replication_times = 20
opt = option_vq.get_args_parser()
fixseed(opt.seed)

opt.device = "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
print(f"Using Device: {opt.device}")

if opt.dataname == 'InterHuman':
    from options.option_data import interh_cfg as data_cfg
    from options.option_data import inter_clip_cfg as eval_model_cfg
    from datasets.interhuman import dataset_VQ, dataset_TM_eval
    from models.evaluator.interhuman.t2m_eval_wrapper import EvaluatorModelWrapper

    ##### ---- Dataloader ---- #####
    data_root = data_cfg.data_root
    eval_wrapper = EvaluatorModelWrapper(eval_model_cfg, data_root, opt.device)
    # ! fix batch_size = eval_model_cfg.batch_size = 96
    val_loader = dataset_TM_eval.DATALoader(data_cfg, True, eval_model_cfg.batch_size)

elif opt.dataname == 'InterX':
    from options.option_data import interx_cfg as data_cfg
    from options.option_data import interx_eval_cfg as eval_model_cfg
    from datasets.interx import dataset_VQ, dataset_TM_eval
    from models.evaluator.interx.t2m_eval_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer

    w_vectorizer = WordVectorizer(pjoin(data_cfg.data_root, 'glove'), 'hhi_vab')
    eval_wrapper = EvaluatorModelWrapper(opt.device, eval_model_cfg.checkpoints_dir)
    val_loader = dataset_TM_eval.DATALoader(data_cfg, True, eval_model_cfg.batch_size, w_vectorizer, unit_length=2**opt.down_t)
else:
    raise ValueError('Unknown dataset')

##### ---- Testing ---- #####
net, nb_iter = load_vq_model(
    opt,
    opt.vq_model_pth,
)
net.eval()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
for i in tqdm(range(replication_times)):
    best_fid, best_div, R_precision, best_matching = eval_t2hh.evaluation_vqvae_test(opt.dataname, val_loader, net, i, eval_wrapper=eval_wrapper)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(R_precision[0])
    top2.append(R_precision[1])
    top3.append(R_precision[2])
    matching.append(best_matching)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)

print(f'\nfinal result: [{nb_iter}]')

msg_final = \
    f"\tFID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tDiversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tTOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(replication_times):.3f}, "\
    f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(replication_times):.3f}, "\
    f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tMatching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(replication_times):.3f}\n\n"
print(msg_final)
