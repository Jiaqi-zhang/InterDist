import copy
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

import torch
import options.option_t2m as option_trans
import models.mask_transformer.t2m_trans as trans

import models.vq.vqvae as vqvae
from utils.utils import fixseed
import utils.eval_t2hh as eval_t2hh


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
    print(f'Loading VQVAE checkpoint from {opt.vq_model_pth} completed!, Nb_iter: [{nb_iter}]')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    net.to(opt.device)
    return net


##### ---- Exp dirs ---- #####
opt = option_trans.get_args_parser(is_train=False)
fixseed(opt.seed)

opt.device = "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
print(f"Using Device: {opt.device}")
torch.autograd.set_detect_anomaly(True)  # Enable PyTorch anomaly detection

unit_length = 2**opt.down_t
if opt.dataname == 'InterHuman':
    from options.option_data import interh_cfg as data_cfg
    from options.option_data import inter_clip_cfg as eval_model_cfg
    from datasets.interhuman import dataset_TM_eval
    from models.evaluator.interhuman.t2m_eval_wrapper import EvaluatorModelWrapper
        
    ##### ---- Dataloader ---- #####
    data_root = data_cfg.data_root
    eval_wrapper = EvaluatorModelWrapper(eval_model_cfg, data_root, opt.device)
    # ! fix batch_size = eval_model_cfg.batch_size = 96
    val_loader = dataset_TM_eval.DATALoader(data_cfg, True, eval_model_cfg.batch_size)
        
elif opt.dataname == 'InterX':
    from options.option_data import interx_cfg as data_cfg
    from options.option_data import interx_eval_cfg as eval_model_cfg
    from datasets.interx import dataset_TM_eval
    from models.evaluator.interx.t2m_eval_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer
        
    ##### ---- Dataloader ---- #####
    w_vectorizer = WordVectorizer(pjoin(data_cfg.data_root, 'glove'), 'hhi_vab')
    eval_wrapper = EvaluatorModelWrapper(opt.device, eval_model_cfg.checkpoints_dir)    
    val_loader = dataset_TM_eval.DATALoader(data_cfg, True, eval_model_cfg.batch_size, w_vectorizer, unit_length=unit_length)

else:
    raise ValueError('Unknown dataset')


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
    # clip_version="ViT-B/32",
    clip_version="ViT-L/14@336px",
)

ckpt = torch.load(opt.resume_trans, map_location='cpu')
missing_keys, unexpected_keys = trans_encoder.load_state_dict(ckpt['t2m_transformer'], strict=False)
# print("missing_keys: " + str(missing_keys))
# print("unexpected_keys: " + str(unexpected_keys))
assert len(unexpected_keys) == 0
assert all([k.startswith('clip_model.') for k in missing_keys])
print(f'Loading Mask Transformer from {opt.resume_trans} completed!, nb_iter {ckpt["nb_iter"]}!')
trans_encoder.eval()
trans_encoder.cuda()


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []

replication_times = 20 
for i in tqdm(range(replication_times)):
    best_fid, best_div, R_precision, best_matching, best_mm = eval_t2hh.evaluation_transformer_mask_test(
        opt.dataname, i, val_loader, net, trans_encoder,
        eval_wrapper=eval_wrapper, cond_scale=opt.cond_scale,
        temperature=opt.temperature, topkr=opt.topkr, force_mask=opt.force_mask,
    )
    fid.append(best_fid)
    div.append(best_div)
    top1.append(R_precision[0])
    top2.append(R_precision[1])
    top3.append(R_precision[2])
    matching.append(best_matching)
    multi.append(best_mm)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)

print(f'\nFinal result: [{ckpt["nb_iter"]}], cond_scale=[{opt.cond_scale}], seed=[{opt.seed}]')
msg_final = \
    f"\tFID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tDiversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tTOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(replication_times):.3f}, "\
    f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(replication_times):.3f}, "\
    f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tMatching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(replication_times):.3f}, \n"\
    f"\tMulti. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(replication_times):.3f}\n\n"
print(msg_final)
