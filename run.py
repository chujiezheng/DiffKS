# coding:utf-8
import argparse
import time
import random
import torch
import numpy as np

from utils import Storage
from main import main

parser = argparse.ArgumentParser(description='A seq2seq model')
args = Storage()

parser.add_argument('--name', type=str, default=None,
    help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
parser.add_argument('--restore', type=str, default=None,
    help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. '
         'Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
parser.add_argument('--mode', type=str, default="train",
    help='"train" or "test". Default: train')
parser.add_argument('--dataset', type=str, default='WizardOfWiki',
    help='Dataloader class. Default: WizardOfWiki')
parser.add_argument('--datapath', type=str, default='./data',
    help='Directory for data set. Default: ./data')
parser.add_argument('--epoch', type=int, default=20,
    help="Epoch for training. Default: 20")
parser.add_argument('--wvclass', type=str, default=None,
    help="Wordvector class, none for not using pretrained wordvec. Default: None")
parser.add_argument('--wvpath', type=str, default=None,
    help="Directory for pretrained wordvector. Default: ./wordvec")


parser.add_argument('--droprate', type=float, default=0.0, help="")
parser.add_argument('--disentangle', action="store_true", help='Disentangle two selectors.')
parser.add_argument('--hist_len', type=int, default=1, help='the number of historical knowledge')
parser.add_argument('--hist_weights', nargs='+', type=float, default=[1.])

parser.add_argument('--out_dir', type=str, default="./output",
    help='Output directory for test output. Default: ./output')
parser.add_argument('--model_dir', type=str, default="./model",
    help='Checkpoints directory for model. Default: ./model')
parser.add_argument('--cache_dir', type=str, default="./cache",
    help='Checkpoints directory for cache. Default: ./cache')
parser.add_argument('--cuda', type=int, default=0, help='Specify the number of gpu to use. Default: 0')
parser.add_argument('--seed', type=int, default=17, help='random seed')
parser.add_argument('--cpu', action="store_true", help='Use cpu.')
parser.add_argument('--debug', action='store_true', help='Enter debug mode (using ptvsd).')
parser.add_argument('--cache', action='store_true',
    help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
cargs = parser.parse_args()


# Editing following arguments to bypass command line.
args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
args.restore = cargs.restore
args.mode = cargs.mode
args.dataset = cargs.dataset
args.datapath = cargs.datapath
args.epochs = cargs.epoch
args.wvclass = cargs.wvclass
args.wvpath = cargs.wvpath
args.out_dir = cargs.out_dir
args.model_dir = cargs.model_dir
args.cache_dir = cargs.cache_dir
args.debug = cargs.debug
args.cache = cargs.cache
args.cuda_num = cargs.cuda
args.cuda = not cargs.cpu

args.disentangle = cargs.disentangle
args.droprate = cargs.droprate
args.hist_len = cargs.hist_len
args.hist_weights = cargs.hist_weights
if args.hist_len != len(args.hist_weights):
    raise ValueError('the hist_len should be equal to the length of weights')
args.hist_weights = np.array(args.hist_weights) / sum(args.hist_weights)

# The following arguments are not controlled by command line.
args.restore_optimizer = False
args.load_exclude_set = []
args.restoreCallback = None

args.batch_num_per_gradient = 1
args.embedding_size = 300
args.eh_size = 200
args.dh_size = 400
args.lr = 5e-4
args.batch_size = 8
args.grad_clip = 5
args.show_sample = [0]  # show which batch when evaluating at tensotboard
args.checkpoint_steps = 3
args.checkpoint_max_to_keep = 3
args.checkpoint_epoch = 5


random.seed(cargs.seed)
np.random.seed(cargs.seed)
torch.manual_seed(cargs.seed)
random.seed(cargs.seed)
torch.cuda.manual_seed(cargs.seed)
torch.cuda.manual_seed_all(cargs.seed)

main(args)
