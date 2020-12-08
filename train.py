from utils.my_trainer import MyTrainer 
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--save_dir', 
                        help='directory to save models.')
    parser.add_argument('--data-dir', default='../../data/SHA2_Bayes',
                        help='training data directory')
    parser.add_argument('--net', default='vgg19',
                        help='network')
    parser.add_argument('--skip-test', default=False,
                        help='no test phase')
    parser.add_argument('--s', type=int, default=5,
                        help='the size of kernel')
    parser.add_argument('--soft', type=bool, default=False,
                        help='use softmax')


    parser.add_argument('--lr', type=float, default=5e-7,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--resize', type=bool, default=False,
                        help='resize image 1024')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = MyTrainer(args)
    trainer.setup()
    trainer.train()
