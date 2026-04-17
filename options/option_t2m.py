import argparse
import os
import torch


def get_args_parser(is_train=False):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader    
    parser.add_argument('--dataname', type=str, default='InterHuman', help='dataset directory')
    parser.add_argument('--batch-size', default=48, type=int, help='batch size')
    parser.add_argument("--gpu_id", type=int, default=3, help='GPU id')
    parser.add_argument('--save_folder', type=str, default='results/', help='result are saved here')
    parser.add_argument('--exp_name', type=str, default='t2m_tmp_debug', help='name of the experiment, will create a file inside out-dir')
    
    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.9, type=float)
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=8192, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=2, help="depth of the network")  # originally 3
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')    
    parser.add_argument('--vq_n_heads', type=int, default=8, help='Number of heads.')
    parser.add_argument('--vq_ff_size', type=int, default=1024, help='FF_Size')
    parser.add_argument('--num_quantizers', type=int, default=1, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')


    ## gpt arch
    parser.add_argument("--clip_dim", type=int, default=768, help="latent dimension in the clip feature")
    parser.add_argument("--embed_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument('--latent_dim', type=int, default=768, help='Dimension of transformer latent.')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of heads.')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of attention layers.')
    parser.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio in transformer')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
    
    ## resume
    parser.add_argument("--vq_model_pth", type=str, default=None, help='resume pth for VQ')    
    parser.add_argument("--resume_trans", type=str, default=None, help='resume gpt pth')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=200, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=10107, type=int, help='seed for initializing training. ')
    
    ## generator
    parser.add_argument("--cond_scale", default=3, type=float, help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    parser.add_argument("--temperature", default=1., type=float, help="Sampling Temperature.")
    parser.add_argument("--topkr", default=0.9, type=float, help="Filter out percentil low prop entries.")
    parser.add_argument('--force_mask', action="store_true", help='True: mask out conditions')
    
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)
    opt.is_train = is_train
    if is_train:
        expr_dir = os.path.join(opt.save_folder, opt.dataname, opt.exp_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
    return parser.parse_args()
