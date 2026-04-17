import os
import json
import copy
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange, einsum

import options.option_t2m as option_trans
from models.vq import vqvae
from models.mask_transformer import t2m_trans as trans
from utils.utils import is_float, is_number, fixseed
import utils.eval_t2hh as eval_t2hh
import utils.utils_model as utils_model


# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):

    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb


def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num * 100 / mask.sum()


def load_vq_model(opt):
    vq_opt = copy.deepcopy(opt)
    vq_opt.motion_dim = data_cfg.motion_dim
    vq_opt.dist_dim = data_cfg.dist_dim
    vq_opt.dim_joint = data_cfg.dim_joint
    vq_opt.nb_joints = data_cfg.nb_joints
    vq_opt.n_heads = vq_opt.vq_n_heads
    vq_opt.ff_size = vq_opt.vq_ff_size
    
    net = vqvae.HumanVQVAE(vq_opt)
    ckpt = torch.load(opt.vq_model_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    nb_iter = ckpt['nb_iter'] if 'nb_iter' in ckpt else -1
    logger.info(f'Loading VQVAE checkpoint from {opt.vq_model_pth} completed!, Nb_iter: [{nb_iter}]')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    net.to(opt.device)
    return net


def update_lr_warm_up(opt_t2m_transformer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in opt_t2m_transformer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


##### ---- Exp dirs ---- #####
opt = option_trans.get_args_parser(is_train=True)
fixseed(opt.seed)

opt.device = "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
print(f"Using Device: {opt.device}")
torch.autograd.set_detect_anomaly(True)  # Enable PyTorch anomaly detection

opt.save_root_dir = pjoin(opt.save_folder, opt.dataname, opt.exp_name)
os.makedirs(opt.save_root_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(opt.save_root_dir)
writer = SummaryWriter(opt.save_root_dir)
logger.info(json.dumps(vars(opt), indent=4, sort_keys=True))

unit_length = 2**opt.down_t
if opt.dataname == 'InterHuman':
    from options.option_data import interh_cfg as data_cfg
    from options.option_data import inter_clip_cfg as eval_model_cfg
    from datasets.interhuman import dataset_TM_eval, dataset_TM_train
    from models.evaluator.interhuman.t2m_eval_wrapper import EvaluatorModelWrapper

    ##### ---- Dataloader ---- #####
    data_root = data_cfg.data_root
    eval_wrapper = EvaluatorModelWrapper(eval_model_cfg, data_root, opt.device)
    # ! fix batch_size = eval_model_cfg.batch_size = 96
    val_loader = dataset_TM_eval.DATALoader(data_cfg, False, eval_model_cfg.batch_size)

    # opt, batch_size, codebook_size, unit_length=4, num_workers=8, shuffle=True, is_training=True
    train_loader = dataset_TM_train.DATALoader(data_cfg, opt.batch_size, is_training=True)
    train_loader_iter = dataset_TM_train.cycle(train_loader)

    # only for eval acc
    val_for_train_loader = dataset_TM_train.DATALoader(data_cfg, opt.batch_size, is_training=False)

elif opt.dataname == 'InterX':
    from options.option_data import interx_cfg as data_cfg
    from options.option_data import interx_eval_cfg as eval_model_cfg
    from datasets.interx import dataset_TM_eval, dataset_TM_train
    from models.evaluator.interx.t2m_eval_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer

    ##### ---- Dataloader ---- #####
    w_vectorizer = WordVectorizer(pjoin(data_cfg.data_root, 'glove'), 'hhi_vab')
    eval_wrapper = EvaluatorModelWrapper(opt.device, eval_model_cfg.checkpoints_dir)
    val_loader = dataset_TM_eval.DATALoader(data_cfg, False, eval_model_cfg.batch_size, w_vectorizer, unit_length=unit_length)

    train_loader = dataset_TM_train.DATALoader(data_cfg, w_vectorizer, opt.batch_size, unit_length=unit_length, is_training=True)
    train_loader_iter = dataset_TM_train.cycle(train_loader)

    # only for eval acc
    val_for_train_loader = dataset_TM_train.DATALoader(data_cfg, w_vectorizer, opt.batch_size, unit_length=unit_length, is_training=False)

else:
    raise ValueError('Unknown dataset')
logger.info(f'Training on {opt.dataname}, motions are with {data_cfg.nb_joints} joints')

##### ---- Network ---- #####
net = load_vq_model(opt)
trans_encoder = trans.MaskTransformer(
    embed_dim=opt.embed_dim,
    device=opt.device,
    num_tokens=opt.nb_code,
    cond_mode='text',
    latent_dim=opt.latent_dim,
    ff_size=opt.ff_size,
    num_layers=opt.n_layers,
    num_heads=opt.n_heads,
    dropout=opt.dropout,
    clip_dim=opt.clip_dim,
    cond_drop_prob=opt.cond_drop_prob,
    clip_version="ViT-L/14@336px",
)

start_nb_iter = 1
if opt.resume_trans is not None:
    ckpt = torch.load(opt.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['t2m_transformer'], strict=False)
    start_nb_iter = ckpt['nb_iter'] if 'nb_iter' in ckpt else 1
    logger.info(f'Loading transformer checkpoint from {opt.resume_trans} completed!, Nb_iter: [{start_nb_iter}]')
trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(opt.decay_option, opt.lr, opt.weight_decay, trans_encoder, opt.optimizer, beta1=opt.beta1, beta2=opt.beta2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_scheduler, gamma=opt.gamma)

##### ---- Training ---- #####
best_fid = 1000
best_iter = 0
best_div = 100
best_top1 = 0
best_top2 = 0
best_top3 = 0
best_matching = 100


for nb_iter in tqdm(range(start_nb_iter, opt.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)

    if nb_iter < opt.warm_up_iter and opt.resume_trans is None:
        current_lr = update_lr_warm_up(optimizer, nb_iter, opt.warm_up_iter, opt.lr)

    clip_text, motion, set_motion, joint_dist, motion_len = batch
    conds = clip_text.cuda().float() if torch.is_tensor(clip_text) else clip_text

    # quants
    motion = motion.cuda().detach().float()
    set_motion = set_motion.cuda().detach().float()
    joint_dist = joint_dist.cuda().detach().float()
    motion_len = motion_len.cuda().long()

    m1_tokens, m2_tokens, d_tokens = net.encode(set_motion, joint_dist)  # m_tokens contains the token corresponding to pad
    m_tokens_len = motion_len // unit_length

    # ce_loss, pred_id, acc, pred_lgm, logits_lm
    loss_cls, pred_acc, logits_lst = trans_encoder(m1_tokens, m2_tokens, d_tokens, conds, m_tokens_len)

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % opt.print_iter == 0:
        writer.add_scalar('Train/cls_loss', loss_cls.item(), nb_iter)
        writer.add_scalar('Train/acc', pred_acc, nb_iter)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], nb_iter)

    if nb_iter == 0 or nb_iter % opt.eval_iter == 0 or nb_iter == opt.total_iter:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_t2hh.evaluation_transformer_mask(
            opt.dataname,
            opt.save_root_dir,
            val_loader,
            net,
            trans_encoder,
            logger,
            writer,
            nb_iter,
            best_fid,
            best_iter,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            eval_wrapper=eval_wrapper,
            cond_scale=opt.cond_scale,
        )

        val_cls_loss = []
        val_acc = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_for_train_loader):
                clip_text, motion, set_motion, joint_dist, motion_len = batch_data

                motion = motion.cuda().detach().float()
                set_motion = set_motion.cuda().detach().float()
                joint_dist = joint_dist.cuda().detach().float()
                motion_len = motion_len.cuda().long()

                m1_tokens, m2_tokens, d_tokens = net.encode(set_motion, joint_dist)  # m_tokens contains the token corresponding to pad
                m_tokens_len = motion_len // unit_length

                loss_cls, pred_acc, _ = trans_encoder(m1_tokens, m2_tokens, d_tokens, conds, m_tokens_len)

                val_cls_loss.append(loss_cls.item())
                val_acc.append(pred_acc)

        print(f"Validation loss:{np.mean(val_cls_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")
        writer.add_scalar('Val/cls_loss', np.mean(val_cls_loss), nb_iter)
        writer.add_scalar('Val/acc', np.mean(val_acc), nb_iter)

    if nb_iter == opt.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)

    # Stop all devices
    if nb_iter == opt.total_iter:
        break
