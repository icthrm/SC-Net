import argparse

import torch.backends.cudnn as cudnn
from train_3DEM import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

# FLAG_PLATFORM = 'laptop'
FLAG_PLATFORM = 'colab'

## setup parse
parser = argparse.ArgumentParser(description='Sparsity Constraint network for Cryo-ET restoration')

if FLAG_PLATFORM == 'colab':
    parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

elif FLAG_PLATFORM == 'laptop':
    parser.add_argument('--gpu_ids', default='-1', dest='gpu_ids')

parser.add_argument('--dir_checkpoint', default='./checkpoint', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--mode', default='train', choices=['train', 'test', 'filter'], dest='mode')
parser.add_argument('--train_continue', default='on', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--norm', type=str, default='bnorm', dest='norm')

parser.add_argument('--name_data', type=str, default='EMNet', dest='name_data')

parser.add_argument('--num_epoch', type=int,  default=5, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=8, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=4e-4, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=2048, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=2048, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=1, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=96, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=96, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=1, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=96, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=96, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=1, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=32, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')

parser.add_argument('--num_freq_disp', type=int,  default=10, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')

# params for SC-Net
parser.add_argument('--dir_data', default='./datasets', dest='dir_data')
parser.add_argument('--dir_test_tomo', default='./PF_Denoised', dest='dir_test_tomo')
parser.add_argument('--tilesize', type=int, default=96, dest='tilesize')
parser.add_argument('--swap_window_size', type=int, default=5, dest='swap_window_size')
parser.add_argument('--swap_mode', type=str, default='internal', choices=['internal', 'cross'], dest='swap_mode')
parser.add_argument('--swap_ratio', type=float, default=0.05, dest='swap_ratio')
parser.add_argument('--swap_region', type=str, default='single', choices=['volume', 'single', 'none'], dest='swap_region')
parser.add_argument('--swap_size', type=int, default=3, dest='swap_size')
parser.add_argument('--padding', type=int, default=32, dest='padding')
parser.add_argument('--N_train', type=int, default=1100, dest='N_train')
parser.add_argument('--N_test', type=int, default=100, dest='N_test')
parser.add_argument('--lambda_exp', type=float, default=0.6, dest='lambda_exp')
parser.add_argument('--lambda_grad', type=float, default=0.2, dest='lambda_grad')
parser.add_argument('--lambda_TV', type=float, default=0.01, dest='lambda_TV')
parser.add_argument('--lambda_median', type=float, default=0.1, dest='lambda_median')

parser = Parser(parser)

def main():
    args = parser.get_arguments()
    parser.write_args()
    parser.print_args()

    trainer = Train(args)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()

if __name__ == '__main__':
    main()
