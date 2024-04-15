import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import os
from train import train

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='weight decay')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Num workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--model', type=str, default="fcn",
                        help='Model name')
    parser.add_argument('--gpus_num', type=str, default="0",
                        help='Model name')
    parser.add_argument('-g', '--gpus', default=True, type=bool,
                        help='number of gpus to use (assumes all are on one node)')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='the rand seed')
    parser.add_argument('--task', type=str, default="spg")

    args = parser.parse_args()
    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    set_seed(seed)

    train(args)