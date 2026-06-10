import argparse
import os
import torch


def get_args_parser(is_train=False):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataname', type=str, default='InterHuman', help='dataset directory')
    parser.add_argument('--motion_rep', type=str, default='smpl', help='how is the motion represented')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    parser.add_argument('--save_folder', type=str, default='results/', help='result are saved here')
    parser.add_argument('--exp_name', type=str, default='tmp_debug', help='name of the experiment, will create a file inside out-dir')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[100000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--loss_explicit', type=float, default=1, help='hyper-parameter for the explicit loss')#0.1
    parser.add_argument('--loss_vel', type=float, default=100, help='hyper-parameter for the velocity loss')
    parser.add_argument('--loss_bn', type=float, default=5, help='hyper-parameter for the bone length loss')
    parser.add_argument('--loss_geo', type=float, default=0.01, help='hyper-parameter for the geodesic loss') #0.001
    parser.add_argument('--loss_fc', type=float, default=500, help='hyper-parameter for the foot contact loss') #100
    parser.add_argument('--loss_dist', type=float, default=10, help='hyper-parameter for the joint dist loss') #100
    parser.add_argument('--ex_loss', action='store_true', help='Enable external loss calculation.')
    
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
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads.')
    parser.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
    
    ## quantizer
    parser.add_argument('--num_quantizers', type=int, default=1, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')

    ## resume
    parser.add_argument("--vq_model_pth", type=str, default=None, help='resume pth for Dist VQ')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=2500, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training.')
    
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)
    opt.is_train = is_train
    if is_train:
        # save to the disk
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
