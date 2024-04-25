import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, drop_rate=0.2, drop_rate_2=0.4):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        if args.task == 'spg':
            drop_rate=0.05
            drop_rate_2=0.05
            self.conv13 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
            self.conv14 = nn.Conv1d(256, 230, kernel_size=1, stride=1)
        elif args.task == 'crysystem':
            drop_rate=0.2
            drop_rate_2=0.4
            self.conv13 = nn.Conv1d(512, 64, kernel_size=3, stride=1, padding=1)
            self.conv14 = nn.Conv1d(64, 7, kernel_size=1, stride=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate_2)
        self.apply(self.weight_init)
        
    @staticmethod
    def weight_init(m):
          nn.init.xavier_uniform_(m.weight)
          
    def forward(self, x):
        x = F.interpolate(x,size=8192,mode='linear', align_corners=False)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv5(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv6(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv7(x)))
        x = self.dropout(x)  
        x = self.pool(F.leaky_relu(self.conv8(x)))
        x = self.dropout(x)    
        x = self.pool(F.leaky_relu(self.conv9(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv10(x)))
        x = self.dropout(x)      
        x = self.pool(F.leaky_relu(self.conv11(x)))
        x = self.dropout(x)   
        x = self.pool(F.leaky_relu(self.conv12(x)))
        x = self.dropout2(x)   
        
        x = self.pool(self.conv13(x))
        x = self.dropout2(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(x, dim=1)
        return x
