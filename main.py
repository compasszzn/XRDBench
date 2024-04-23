import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import os
from train import train
import datetime
import wandb



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
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--model', type=str, default="cnn2",
                        help='Model name')
    parser.add_argument('--seed', type=int, default=2023, metavar='N',
                        help='the rand seed')
    parser.add_argument('--task', type=str, default="spg")
    
    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='weight decay')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Num workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for early stopping')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    set_seed(seed)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="XRDBench",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )
    train(args)
    wandb.finish()

# sweep_config = {
#     'method': 'random',
#     'metric': {
#     'name': 'val_loss',
#     'goal': 'minimize'   
#     },
#     'parameters':{
#     'learning_rate': {
#         # a flat distribution between 0 and 0.1
#         'distribution': 'uniform',
#         'min': 0,
#         'max': 0.1
#       }
#     }
#     }
# sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")
# wandb.agent(sweep_id, function=main, count=10)