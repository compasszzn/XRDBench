import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from model.fcn import FCN
from model import MIC_CNN2
from model import MIC_CNN3 
from model import ICNN
from model import PQNET
from model import ICSD
from model import MP
from model import AutoAnalyzer
from model import XCA
from model import IUCrJ_CNNspg
from dataset.dataset import ASEDataset
from tqdm import tqdm
import time
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import os
from utils.tools import EarlyStopping
import json
import wandb
import datetime
import warnings


def train(args,test_loss_list, macro_f1_list, macro_precision_list,macro_recall_list,test_accuracy_list):
    warnings.filterwarnings("ignore")
    if args.use_gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    # t1=time.time()
    # train_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/train_1/binxrd.db']
    # val_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/val_db/test_binxrd.db']
    # test_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/test_db/binxrd.db']
    # t2=time.time()
    # inter = t2 - t1
    # print(inter)

    train_dataset = ASEDataset(['/data/zzn/xrdsim/train_1/binxrd.db','/data/zzn/xrdsim/train_2/binxrd.db'],False)
    val_dataset = ASEDataset(['/data/zzn/xrdsim/val_db/test_binxrd.db'],False)
    test_dataset = ASEDataset(['/data/zzn/xrdsim/test_db/binxrd.db'],False)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)

    if args.model == 'cnn2':
        model = MIC_CNN2.Model(args)
    elif args.model == 'cnn3':
        model = MIC_CNN3.Model(args)
    elif args.model == 'icnn':
        model = ICNN.Model(args)
    elif args.model == 'pqnet':
        model = PQNET.Model(args)
    elif args.model == 'icsd':
        model = ICSD.Model(args)
    elif args.model == 'mp':
        model = MP.Model(args)
    elif args.model == 'autoanalyzer':
        model = AutoAnalyzer.Model(0.7,args)
    elif args.model == 'xca':
        model = XCA.Model(args)
    elif args.model == 'IUCrj_CNN':
        model = IUCrJ_CNNspg.Model(args)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # save_path = f'./checkpoints/{args.task}-{args.model}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_{nowtime}'
    # if not os.path.exists('./checkpoints'):
    #     os.mkdir('./checkpoints')
    save_path = f'/data/zzn/checkpoints/{args.task}-{args.model}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_{nowtime}'
    if not os.path.exists('/data/zzn/checkpoints'):
        os.mkdir('/data/zzn/checkpoints')
    os.makedirs(save_path, exist_ok=True)
    with open(save_path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    model = model.to(device)
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_test_accuracy = 0
    best_test_precision = 0
    best_test_recall = 0
    best_test_macrof1=0
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, optimizer, criterion, epoch, train_loader, device, args)
        val_loss, _ = run_epoch(model, optimizer, criterion, epoch, val_loader,device, args, backprop=False)
        test_loss, res = run_epoch(model, optimizer, criterion, epoch, test_loader,device, args, backprop=False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_test_accuracy = res['accuracy']
            best_test_precision = res['macro_precision']
            best_test_recall = res['macro_recall']
            best_test_macrof1=res['macro_f1']
            best_epoch = epoch
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, 
                   "test_loss":test_loss, "macro_f1":res['macro_f1'],
                   "macro_precision":res['macro_precision'],
                   "macro_recall":res['macro_recall'],"test_acc": res['accuracy']})
        early_stopping(val_loss, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": best_val_loss, 
    #                "test_loss":best_test_loss, "test_f1": best_test_macrof1,"test_precision":best_test_precision , 
    #                "test_recall":best_test_recall ,"test_acc": best_test_accuracy})
    test_loss_list.append(best_test_loss)
    macro_f1_list.append(best_test_macrof1)
    macro_precision_list.append(best_test_precision)
    macro_recall_list.append(best_test_recall)
    test_accuracy_list.append(best_test_accuracy)


    print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test Accuracy: %.5f  \t Best Macro-F1: %.5f\t Best epoch %d" %
                (best_val_loss, best_test_loss, best_test_accuracy,best_test_macrof1, best_epoch))

    return best_test_accuracy

def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()
    res = {'epoch': epoch, 'loss': 0, 'accuracy': 0, 'macro_f1': 0 ,'counter': 0,'macro_precision':0,"macro_recall":0}
    all_labels = []
    all_predicted = []
    for batch_index, data in enumerate(tqdm(loader)):
        intensity,crysystem_labels,spg_labels,element = data['intensity'].to(device), data['crysystem'].to(device), data['spg'].to(device), data['element'].to(device)
        intensity = intensity.unsqueeze(1)
        element = element.unsqueeze(1)
        if args.task=='spg':
            labels=spg_labels-1
        elif args.task=='crysystem':
            labels=crysystem_labels-1
        batch_size=intensity.shape[0]
        if backprop:
            optimizer.zero_grad()
            outputs = model(intensity)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = model(intensity)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        # accuracy = (predicted == labels).sum().item()
        # res['accuracy'] += accuracy
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predicted)
    macro_f1 = f1_score(all_labels, all_predicted, average='macro')
    macro_precision = precision_score(all_labels, all_predicted, average='macro')
    macro_recall = recall_score(all_labels, all_predicted, average='macro')

    res['accuracy'] = accuracy
    res['macro_f1'] = macro_f1
    res['macro_precision'] = macro_precision
    res['macro_recall'] = macro_recall   

    if not backprop:
        prefix = "==> "
    else:
        prefix = " "
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'] , res