import math
import torch.nn as nn
    
class RMRNet(nn.Module):
    def __init__(self):
        super(RMRNet, self).__init__()
        # First part
        self.convs = nn.Sequential(nn.Conv2d(10, 32, 3, 2), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 64, 3, 2), nn.BatchNorm2d(64),
                                   nn.Conv2d(64, 128, 3, 2), nn.BatchNorm2d(128),
                                   nn.Conv2d(128, 32, 1, 1), nn.BatchNorm2d(32))
        self.drop = nn.Dropout2d()
        # Second part
        self.lstm = nn.LSTM(1568, 1568, bidirectional=False, num_layers=3)
        # Third part
        self.fc1 = nn.Linear(1568, 512)
        self.branch1 = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        self.branch2 = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        self.branch3 = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        self.branch4 = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        self.branch5 = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        self.relu = nn.Tanh()
        self.p1, self.p2, self.p3, self.p4, self.p5 = 72, 54, 30, 0.2, 0.2
        
    def forward(self, x):
        b, _, h, w = x.size()
        assert h == 64 and w == 64, print("input size must be 64x64")
        x = self.convs(x)
        x = self.drop(x)
        x = x.view(b, -1).unsqueeze(0)  # [1, b, c*h*w]
        x, _ = self.lstm(x)
        x = self.fc1(x.squeeze(0))
        e1 = self.relu(self.branch1(x))
        e2 = self.relu(self.branch2(x))
        e3 = self.relu(self.branch3(x))
        e4 = self.relu(self.branch4(x))
        e5 = self.relu(self.branch5(x))
        dx = e1*self.p1
        dy = e2*self.p2
        theta = e3*self.p3*math.pi/180
        sx = 1 + e4*self.p4
        sy = 1 + e5*self.p5
        return dx, dy, theta, sx, sy # [B, 1]
    
            
            