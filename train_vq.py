import os
import json
from tqdm import tqdm
from os.path import join as pjoin
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import options.option_vq as option_vq
from models.vq.vqvae import HumanVQVAE
from models.losses import Geometric_Losses
from utils.utils import fixseed
from utils import eval_t2hh as eval_t2hh
from utils import utils_model as utils_model


def update_loss_dict(loss_dict, **losses):
    for loss_name, loss_value in losses.items():
        loss_dict[loss_name] += loss_value
    return loss_dict


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


##### ---- Exp dirs ---- #####
opt = option_vq.get_args_parser(is_train=True)
fixseed(opt.seed)

opt.device = "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
print(f"Using Device: {opt.device}")

print(opt.exp_name)
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
    from datasets.interhuman import dataset_VQ, dataset_TM_eval
    from models.evaluator.interhuman.t2m_eval_wrapper import EvaluatorModelWrapper

    ##### ---- Dataloader ---- #####
    data_root = data_cfg.data_root
    eval_wrapper = EvaluatorModelWrapper(eval_model_cfg, data_root, opt.device)
    train_loader = dataset_VQ.DATALoader(data_cfg, opt.batch_size, opt.window_size)
    train_loader_iter = dataset_VQ.cycle(train_loader)
    # ! fix batch_size = eval_model_cfg.batch_size = 96
    val_loader = dataset_TM_eval.DATALoader(data_cfg, False, eval_model_cfg.batch_size)

elif opt.dataname == 'InterX':
    from options.option_data import interx_cfg as data_cfg
    from options.option_data import interx_eval_cfg as eval_model_cfg
    from datasets.interx import dataset_VQ, dataset_TM_eval
    from models.evaluator.interx.t2m_eval_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer

    ##### ---- Dataloader ---- #####
    train_loader = dataset_VQ.DATALoader(data_cfg, opt.batch_size, opt.window_size)
    train_loader_iter = dataset_VQ.cycle(train_loader)

    w_vectorizer = WordVectorizer(pjoin(data_cfg.data_root, 'glove'), 'hhi_vab')
    eval_wrapper = EvaluatorModelWrapper(opt.device, eval_model_cfg.checkpoints_dir)
    val_loader = dataset_TM_eval.DATALoader(data_cfg, False, eval_model_cfg.batch_size, w_vectorizer, unit_length=unit_length)

else:
    raise ValueError('Unknown dataset')
logger.info(f'Training on {opt.dataname}, motions are with {data_cfg.nb_joints} joints')

##### ---- Network ---- #####
opt.motion_dim = data_cfg.motion_dim
opt.dist_dim = data_cfg.dist_dim
opt.dim_joint = data_cfg.dim_joint
opt.nb_joints = data_cfg.nb_joints
net = HumanVQVAE(opt)

start_nb_iter = 1
if opt.vq_model_pth:
    logger.info('loading checkpoint from {}'.format(opt.vq_model_pth))
    ckpt = torch.load(opt.vq_model_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    start_nb_iter = ckpt['nb_iter'] if 'nb_iter' in ckpt else 1
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
norm_data_root = pjoin(data_cfg.data_root, 'motions_processed') if opt.dataname == 'InterHuman' else data_cfg.meta_dir
Loss_geo = Geometric_Losses(norm_data_root, opt.recons_loss, data_cfg.nb_joints, opt.dataname, opt.device, opt.ex_loss)

optimizer = optim.AdamW(net.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_scheduler, gamma=opt.gamma)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in tqdm(range(start_nb_iter, opt.warm_up_iter)):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, opt.warm_up_iter, opt.lr)

    in_motion, out_motion, joint_dist = next(train_loader_iter)
    in_motion = in_motion.cuda().float()
    out_motion = out_motion.cuda().float()
    joint_dist = joint_dist.cuda().float()

    pred_motion1, pred_motion2, loss_commit, perplexity = net(in_motion, joint_dist)

    loss_rec_lst = 0.
    gt_pos_lst, pred_pos_lst = [], []
    loss = opt.commit * loss_commit
    for pred, gt in zip([pred_motion1, pred_motion2], [out_motion[..., :opt.motion_dim // 2], out_motion[..., opt.motion_dim // 2:]]):
        if opt.ex_loss:
            loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, gt_pos, pred_pos = Loss_geo.forward(pred, gt)
            loss = loss + loss_rec + (opt.loss_explicit * loss_explicit) + (opt.loss_vel * loss_vel) + \
                (opt.loss_bn * loss_bn) + (opt.loss_geo * loss_geo) + (opt.loss_fc * loss_fc)

            gt_pos_lst.append(gt_pos)
            pred_pos_lst.append(pred_pos)
        else:
            loss_rec = Loss_geo.forward(pred, gt, only_rec=True)
            loss = loss + loss_rec
        loss_rec_lst += loss_rec.item()

    if opt.ex_loss and opt.dataname == 'InterHuman':
        loss_dist = Loss_geo.calc_loss_dist_interH(
            [pred_motion1, pred_motion2],
            [out_motion[..., :opt.motion_dim // 2], out_motion[..., opt.motion_dim // 2:]],
        )
        loss = loss + loss_dist
    elif opt.ex_loss and opt.dataname == 'InterX':
        loss_dist = Loss_geo.calc_loss_dist_interX(pred_pos_lst, gt_pos_lst)
        loss = loss + loss_dist * opt.loss_dist
    else:
        loss = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_rec_lst
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % opt.print_iter == 0:
        avg_recons /= opt.print_iter
        avg_perplexity /= opt.print_iter
        avg_commit /= opt.print_iter

        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t MCommit. {avg_commit:.5f} \t MPPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_losses_dict = {
    "perplexity": 0.,
    "commit": 0.,
    "loss": 0.,
    "loss_rec": 0.,
    "loss_explicit": 0.,
    "loss_vel": 0.,
    "loss_bn": 0.,
    "loss_geo": 0.,
    "loss_fc": 0.,
    "loss_dist": 0.,
}

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_t2hh.evaluation_vqvae(
    opt.dataname,
    opt.save_root_dir,
    val_loader,
    net,
    logger,
    writer,
    0,
    best_fid=1000,
    best_iter=0,
    best_div=100,
    best_top1=0,
    best_top2=0,
    best_top3=0,
    best_matching=100,
    eval_wrapper=eval_wrapper,
)

for nb_iter in tqdm(range(start_nb_iter, opt.total_iter + 1)):
    in_motion, out_motion, joint_dist = next(train_loader_iter)
    in_motion = in_motion.cuda().float()
    out_motion = out_motion.cuda().float()
    joint_dist = joint_dist.cuda().float()

    pred_motion1, pred_motion2, loss_commit, perplexity = net(in_motion, joint_dist)

    gt_pos_lst, pred_pos_lst = [], []
    loss = opt.commit * loss_commit
    for pred, gt in zip([pred_motion1, pred_motion2], [out_motion[..., :opt.motion_dim // 2], out_motion[..., opt.motion_dim // 2:]]):
        if opt.ex_loss:
            loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, gt_pos, pred_pos = Loss_geo.forward(pred, gt)
            loss = loss + loss_rec + (opt.loss_explicit * loss_explicit) + (opt.loss_vel * loss_vel) + \
                (opt.loss_bn * loss_bn) + (opt.loss_geo * loss_geo) + (opt.loss_fc * loss_fc)
            gt_pos_lst.append(gt_pos)
            pred_pos_lst.append(pred_pos)

            avg_losses_dict = update_loss_dict(
                avg_losses_dict,
                loss_rec=loss_rec.item(),
                loss_explicit=loss_explicit.item(),
                loss_vel=loss_vel.item(),
                loss_bn=loss_bn.item(),
                loss_geo=loss_geo.item(),
                loss_fc=loss_fc.item(),
            )
        else:
            loss_rec = Loss_geo.forward(pred, gt, only_rec=True)
            loss = loss + loss_rec
            avg_losses_dict = update_loss_dict(avg_losses_dict, loss_rec=loss_rec.item())

    if opt.ex_loss and opt.dataname == 'InterHuman':
        loss_dist = Loss_geo.calc_loss_dist_interH(
            [pred_motion1, pred_motion2],
            [out_motion[..., :opt.motion_dim // 2], out_motion[..., opt.motion_dim // 2:]],
        )
        loss = loss + loss_dist
        avg_losses_dict = update_loss_dict(avg_losses_dict, loss_dist=loss_dist.item())
    elif opt.ex_loss and opt.dataname == 'InterX':
        loss_dist = Loss_geo.calc_loss_dist_interX(pred_pos_lst, gt_pos_lst)
        loss = loss + loss_dist * opt.loss_dist
        avg_losses_dict = update_loss_dict(avg_losses_dict, loss_dist=loss_dist.item())
    else:
        loss = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    avg_losses_dict = update_loss_dict(
        avg_losses_dict,
        perplexity=perplexity.item(),
        commit=loss_commit.item(),
        loss=loss.item(),
    )

    if nb_iter % opt.print_iter == 0:
        for loss_name, loss_value in avg_losses_dict.items():
            avg_losses_dict[loss_name] = loss_value / opt.print_iter
            writer.add_scalar(f'Train/{loss_name}', loss_value, nb_iter)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], nb_iter)

        logger.info(f"Train. Iter {nb_iter} : \t MCommit. {avg_losses_dict['commit']:.5f} \t MPPL. {avg_losses_dict['perplexity']:.2f} \t Recons.  {avg_losses_dict['loss_rec']:.5f}")

        avg_losses_dict = {
            "perplexity": 0.,
            "commit": 0.,
            "loss": 0.,
            "loss_rec": 0.,
            "loss_explicit": 0.,
            "loss_vel": 0.,
            "loss_bn": 0.,
            "loss_geo": 0.,
            "loss_fc": 0.,
            "loss_dist": 0.,
        }

    if nb_iter % opt.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_t2hh.evaluation_vqvae(
            opt.dataname,
            opt.save_root_dir,
            val_loader,
            net,
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
        )
