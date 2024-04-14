import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from model.fcn import FCN
from dataset.dataset import ASEDataset
from tqdm import tqdm
import time
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

    train_dataset = ASEDataset('/data/zzn/xrdsim/train_1/binxrd.db', split='train')
    val_dataset = ASEDataset('/data/zzn/xrdsim/val_db/test_binxrd.db', split='val')
    test_dataset = ASEDataset('/data/zzn/xrdsim/test_db/binxrd.db', split='test')



    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    # for batch in test_loader:
    #     print(batch['latt_dis'], batch['intensity'], batch['spg'], batch['crysystem'], batch['split'])

    # Initialize the model
    model = FCN(drop_rate=0.2, drop_rate_2=0.4)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_test_accuracy = 0
    test_interval = 10
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, optimizer, criterion, epoch, train_loader, device, args)
        if epoch % test_interval == 0 or epoch == args.epochs-1:
            val_loss, val_accuracy = run_epoch(model, optimizer, criterion, epoch, val_loader,device, args, backprop=False)
            test_loss, test_accuracy = run_epoch(model, optimizer, criterion, epoch, test_loader,device, args, backprop=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_test_accuracy = test_accuracy
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test Accuracy: %.5f \t Best epoch %d" %
                (best_val_loss, best_test_loss, best_test_accuracy, best_epoch))

            if epoch - best_epoch > 100:
                break

    return best_test_accuracy
def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()
    res = {'epoch': epoch, 'loss': 0, 'accuracy': 0 ,'counter': 0}
    for batch_index, data in enumerate(tqdm(loader)):
        inputs, labels = data['intensity'].to(device), data['spg'].to(device)
        inputs = inputs.unsqueeze(1)
        batch_size=inputs.shape[0]
        if backprop:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item()
        res['accuracy'] += accuracy
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
    if not backprop:
        prefix = "==> "
    else:
        prefix = " "
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'] , res['accuracy']/res['counter']