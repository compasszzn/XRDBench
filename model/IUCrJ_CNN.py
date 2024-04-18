import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv1d(1, 80, kernel_size=100, stride=5, padding=20050)
        self.dropout1 = nn.Dropout(0.3)
        self.avg_pool1 = nn.AvgPool1d(kernel_size=3, stride=2) # pooling layer
        self.cnn2 = nn.Conv1d(80, 80, kernel_size=50, stride=5, padding=10023)
        self.dropout2 = nn.Dropout(0.3)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.cnn3 = nn.Conv1d(80, 80, kernel_size=25, stride=2, padding=2511)
        self.dropout3 = nn.Dropout(0.3)
        self.avg_pool3 = nn.AvgPool1d(kernel_size=3, stride=1)
        
        mlp_in_features = 4996*80
        if args.task == 'spg':
            self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(mlp_in_features, 2300),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(2300, 1150),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(1150, 230))
        elif args.task == 'crysystem':
            self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(mlp_in_features, 700),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(700, 70),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(70, 7))
        
    def forward(self, x):
        x = F.interpolate(x,size=10001,mode='linear', align_corners=False)
        x = F.relu(self.cnn1(x))
        x = self.dropout1(x)
        x = self.avg_pool1(x)
        x = F.relu(self.cnn2(x))
        
        x = self.dropout2(x)
        x = self.avg_pool2(x)
        x = F.relu(self.cnn3(x))
        x = self.dropout3(x)
        x = self.avg_pool3(x)
        
        x = self.MLP(x)
        x = F.softmax(x, dim=1)
        return x
        
        
        
        
        