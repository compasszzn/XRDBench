import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import os
from train import train
import datetime
import wandb
import json



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
    parser.add_argument('--model', type=str, default="FCN",
                        help='Model name')
    parser.add_argument('--seed', type=int, default=300, metavar='N',
                        help='the rand seed')
    parser.add_argument('--task', type=str, default="crysystem")
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for early stopping')
    parser.add_argument('--warmup-epochs', default=2, type=int, metavar='N',
                        help='number of warmup epochs')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=6, help='gpu')

    args = parser.parse_args()
    

    model_result = {'name': args.model,'task':args.task}

    # test_loss_list, macro_f1_list, macro_precision_list,macro_recall_list,test_accuracy_list = [], [], [], [], []
    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    set_seed(seed)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="Official_XRDBench",
        entity="xrdbench",
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )
    output=train(args,nowtime)
    # test_loss_list.append(output.best_test_loss)
    # macro_f1_list.append(output.best_test_macrof1)
    # macro_precision_list.append(output.best_test_precision)
    # macro_recall_list.append(output.best_test_recall)
    # test_accuracy_list.append(output.best_test_accuracy)
    wandb.finish()
    # model_result['loss mean'] = np.mean(test_loss_list)
    # model_result['loss std'] = np.std(test_loss_list)
    # model_result['f1 mean'] = np.mean(macro_f1_list)
    # model_result['f1 std'] = np.std(macro_f1_list)
    # model_result['precision mean'] = np.mean(macro_precision_list)
    # model_result['precision std'] = np.std(macro_precision_list)
    # model_result['recall mean'] = np.mean(macro_recall_list)
    # model_result['recall std'] = np.std(macro_recall_list)
    # model_result['accuracy mean'] = np.mean(test_accuracy_list)
    # model_result['accuracy std'] = np.std(test_accuracy_list)
    # save_path = f'./output'
    # if not os.path.exists('./output'):
    #     os.mkdir('./output')
    # os.makedirs(save_path, exist_ok=True)
    # with open(save_path+f"/{args.model}_{args.task}.json", "w") as f:
    #     json.dump(model_result, f)
    # print(model_result)
    