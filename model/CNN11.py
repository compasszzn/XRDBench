from torch import nn
import torch.nn.functional as F
class NoPoolCNN(nn.Module):
    def __init__(self):
        super(NoPoolCNN, self).__init__()
        self.CNN = nn.Sequential(
                nn.Conv1d(1, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, x):
        return self.CNN(x)

class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, out_features))

    def forward(self, x):
        return self.MLP(x)
    

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.cnn = NoPoolCNN()
        mlp_in_features = 12160
        if args.task == 'spg':
            self.MLP = Predictor(mlp_in_features, 230)
        elif args.task == 'crysystem':
            self.MLP = Predictor(mlp_in_features, 7)
        
    def forward(self, x):
        x = F.interpolate(x,size=8500,mode='linear', align_corners=False)
        x = self.cnn(x)
        x = self.MLP(x)
        return x