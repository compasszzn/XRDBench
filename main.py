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
    parser.add_argument('--model', type=str, default="cnn3",
                        help='Model name')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='the rand seed')
    parser.add_argument('--task', type=str, default="crysystem")
    
    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='weight decay')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Num workers in dataloader')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for early stopping')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=3, help='gpu')

    args = parser.parse_args()
    

    model_result = {'name': args.model}

    test_loss_list, macro_f1_list, macro_precision_list,macro_recall_list,test_accuracy_list = [], [], [], [], []
    for t in range(args.trials):
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
        train(args,test_loss_list, macro_f1_list, macro_precision_list,macro_recall_list,test_accuracy_list)
        wandb.finish()
    model_result['loss mean'] = np.mean(test_loss_list)
    model_result['loss std'] = np.std(test_loss_list)
    model_result['f1 mean'] = np.mean(macro_f1_list)
    model_result['f1 std'] = np.std(macro_f1_list)
    model_result['precision mean'] = np.mean(macro_precision_list)
    model_result['precision std'] = np.std(macro_precision_list)
    model_result['recall mean'] = np.mean(macro_recall_list)
    model_result['recall std'] = np.std(macro_recall_list)
    model_result['accuracy mean'] = np.mean(test_accuracy_list)
    model_result['accuracy std'] = np.std(test_accuracy_list)
    print(model_result)