import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from model.fcn import FCN
from model.cnn2 import CNN2
from model.cnn3 import CNN3
from model.cnn import CNN
from model.pqnet import PQNET
from model.icsd import ICSD
from model.mp import MP
from model.xca import XCA
from model.autoanalyzer import AUTOANALYZER
from dataset.dataset import ASEDataset
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
def train(args):

    if args.gpus:
        device = torch.device('cuda:' + str(args.gpus_num))
    else:
        device = 'cpu'
    # t1=time.time()
    # train_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/train_1/binxrd.db']
    # val_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/val_db/test_binxrd.db']
    # test_file_paths = ['/hpc2hdd/home/zzheng078/jhspoolers/xrdsim/test_db/binxrd.db']
    # t2=time.time()
    # inter = t2 - t1
    # print(inter)

    train_dataset = ASEDataset(['/data/zzn/xrdsim/train_1/binxrd.db','/data/zzn/xrdsim/train_2/binxrd.db'])
    val_dataset = ASEDataset(['/data/zzn/xrdsim/val_db/test_binxrd.db'])
    test_dataset = ASEDataset(['/data/zzn/xrdsim/test_db/binxrd.db'])



    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # for batch in test_loader:
    #     print(batch['latt_dis'], batch['intensity'], batch['spg'], batch['crysystem'], batch['split'])

    # Initialize the model
    # if args.model == 'fcn':
    #     model = FCN(drop_rate=0.2, drop_rate_2=0.4,task=args.task)
    if args.model == 'cnn2':
        model = CNN2(task=args.task)
    elif args.model == 'cnn3':
        model = CNN3(task=args.task)
    elif args.model == 'cnn':
        model = CNN(task=args.task)
    elif args.model == 'pqnet':
        model = PQNET(task=args.task)
    elif args.model == 'icsd':
        model = ICSD(task=args.task)
    elif args.model == 'mp':
        model = MP(task=args.task)
    elif args.model == 'autoanalyzer':
        model = AUTOANALYZER(dropout_rate=0.7,task=args.task)
    elif args.model == 'xca':
        model = XCA(task=args.task)

    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_test_accuracy = 0
    best_test_microf1=0
    best_test_macrof1=0
    test_interval = 1
    for epoch in range(args.epochs):
        train_loss, train_accuracy,_ = run_epoch(model, optimizer, criterion, epoch, train_loader, device, args)
        if epoch % test_interval == 0 or epoch == args.epochs-1:
            val_loss, val_accuracy,_ = run_epoch(model, optimizer, criterion, epoch, val_loader,device, args, backprop=False)
            test_loss, test_accuracy,res = run_epoch(model, optimizer, criterion, epoch, test_loader,device, args, backprop=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_test_accuracy = test_accuracy
                best_test_microf1=res['micro_f1']
                best_test_macrof1=res['macro_f1']
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test Accuracy: %.5f \t Best Micro-F1: %.5f \t Best Macro-F1: %.5f\t Best epoch %d" %
                (best_val_loss, best_test_loss, best_test_accuracy,best_test_microf1,best_test_macrof1, best_epoch))

            if epoch - best_epoch > 100:
                break

    return best_test_accuracy
def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()
    res = {'epoch': epoch, 'loss': 0, 'accuracy': 0, 'micro_f1': 0, 'macro_f1': 0 ,'counter': 0}
    all_labels = []
    all_predicted = []
    for batch_index, data in enumerate(tqdm(loader)):
        intensity, latt_dis,crysystem_labels,spg_labels = data['intensity'].to(device),data['latt_dis'].to(device), data['crysystem'].to(device), data['spg'].to(device)
        intensity = intensity.unsqueeze(1)
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
        accuracy = (predicted == labels).sum().item()
        res['accuracy'] += accuracy
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    micro_f1 = f1_score(all_labels, all_predicted, average='micro')
    macro_f1 = f1_score(all_labels, all_predicted, average='macro')

    res['micro_f1'] = micro_f1
    res['macro_f1'] = macro_f1
    if not backprop:
        prefix = "==> "
    else:
        prefix = " "
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'] , res['accuracy']/res['counter'], res