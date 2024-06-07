import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1,CNN2,CNN3,CNN4,CNN5,CNN6,CNN7,CNN8,CNN9,CNN10,CNN11, MLP, GRU, LSTM, RNN, BiGRU, BiLSTM, BiRNN, Transformer, iTransformer, PatchTST
from dataset.dataset import ASEDataset
from tqdm import tqdm
import time
import wget
import sys
from dataset.parse import load_dataset,bar_progress
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import os
from utils.tools import EarlyStopping, adjust_learning_rate_withWarmup
import json
import wandb
import datetime
import warnings
time_exp_dic = {'time': 0, 'counter': 0}

def train(args,nowtime):
    warnings.filterwarnings("ignore")
    if args.use_gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # 1 : Dataset in DB Files
    train_dataset = ASEDataset([os.path.join(args.datapath, 'train_1/binxrd.db'),os.path.join(args.datapath, 'train_2/binxrd.db')],False)
    val_dataset = ASEDataset([os.path.join(args.datapath, 'val_db/test_binxrd.db')],False)
    test_dataset = ASEDataset([os.path.join(args.datapath, 'test_db/binxrd.db')],False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
    
    """
    2 : Dataset in TFRecord Files: A Small Dataset for Review
    # 1. Point to a local or remote Croissant file
    # import mlcroissant as mlc
    url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

    # 2. Inspect metadata
    dataset_info = mlc.Dataset(url).metadata.to_json
    print(dataset_info)

    for file_info in dataset_info['distribution']:
        wget.download(file_info['contentUrl'], './', bar=bar_progress)

    # 3. Use Croissant dataset in your ML workload
    from dataset.parse import load_dataset,bar_progress # defined in our github
    train_loader = DataLoader(load_dataset(name='train.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(load_dataset(name='val.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=False)
    test_loader = DataLoader(load_dataset(name='test.tfrecord'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
    """

    if args.model == 'cnn1':
        model = CNN1.Model(args)
    elif args.model == 'cnn2':
        model = CNN2.Model(args)
    elif args.model == 'cnn3':
        model = CNN3.Model(args)
    elif args.model == 'cnn4':
        model = CNN4.Model(args)
    elif args.model == 'cnn5':
        model = CNN5.Model(args)
    elif args.model == 'cnn6':
        model = CNN6.Model(args)
    elif args.model == 'cnn7':
        model = CNN7.Model(0.7,args)
    elif args.model == 'cnn8':
        model = CNN8.Model(args)
    elif args.model == 'cnn9':
        model = CNN9.Model(args)
    elif args.model == 'cnn10':
        model = CNN10.Model(args)
    elif args.model == 'cnn11':
        model = CNN11.Model(args)
    elif args.model == 'mlp':
        model = MLP.Model(args)
    elif args.model == 'rnn':
        model = RNN.Model(args)
    elif args.model == 'lstm':
        model = LSTM.Model(args)
    elif args.model == 'gru':
        model = GRU.Model(args)
    elif args.model == 'birnn':
        model = BiRNN.Model(args)
    elif args.model == 'bilstm':
        model = BiLSTM.Model(args)
    elif args.model == 'bigru':
        model = BiGRU.Model(args)
    elif args.model == 'transformer':
        model = Transformer.Model(args)
    elif args.model == 'iTransformer':
        model = iTransformer.Model(args)
    elif args.model == 'PatchTST':
        model = PatchTST.Model(args)

    save_path = f'./checkpoints/{args.task}-{args.model}_lr{args.lr}_bs{args.batch_size}_{nowtime}_{args.seed}'
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    os.makedirs(save_path, exist_ok=True)
    with open(save_path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, optimizer, criterion, epoch, train_loader, device, args)
        val_loss, _ = run_epoch(model, optimizer, criterion, epoch, val_loader,device, args, backprop=False)
        test_loss, res = run_epoch(model, optimizer, criterion, epoch, test_loader,device, args, backprop=False)

        wandb.log({"epoch": epoch, "val_loss": val_loss, 
                   "test_loss":test_loss, "test_f1":res['macro_f1'],
                   "test_precision":res['macro_precision'],
                   "test_recall":res['macro_recall'],"test_acc": res['accuracy'],"infer_time":time_exp_dic['time'] / time_exp_dic['counter']})
        early_stopping(epoch,val_loss,test_loss,res, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    wandb.log({"epoch": early_stopping.best_epoch, "val_loss": early_stopping.val_loss_min, 
                   "test_loss":early_stopping.best_test_loss, "test_f1": early_stopping.best_test_macrof1,
                   "test_precision":early_stopping.best_test_precision , 
                   "test_recall":early_stopping.best_test_recall ,"test_acc": early_stopping.best_test_accuracy})



    print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test Accuracy: %.5f  \t Best Macro-F1: %.5f\t Best epoch %d" %
                (early_stopping.val_loss_min, early_stopping.best_test_loss, early_stopping.best_test_accuracy,early_stopping.best_test_macrof1, early_stopping.best_epoch))

    return early_stopping

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

            torch.cuda.synchronize()
            t1 = time.time()
            outputs = model(intensity)
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = model(intensity)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)

        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
        if args.model == 'CPICANN':
            adjust_learning_rate_withWarmup(optimizer, epoch + batch_index / len(loader), args)

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