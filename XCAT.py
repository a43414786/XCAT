import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import Conv2d, PixelShuffle, ReLU

class HXBlock(nn.Module):
    def __init__(self) -> None:
        super(HXBlock, self).__init__()
        self.cnn_1 = Conv2d(7,7,(3,3),(1,1),(1,1))
        self.cnn_2 = Conv2d(21,21,(1,1),(1,1),(0,0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.cnn_1(x[:,:7,:,:])
        X = self.cnn_2(x[:,7:,:,:])
        out = torch.cat((X,H),dim=1)
        return out

class XCAT(nn.Module):
    def __init__(self,m) -> None:
        super(XCAT, self).__init__()
        self.m = m 
        self.cnn_1 = Conv2d(3,28,(3,3),(1,1),(1,1))
        self.cnn_2 = Conv2d(28,27,(3,3),(1,1),(1,1))
        self.cnn_3 = Conv2d(27,27,(3,3),(1,1),(1,1))
        self.relu = ReLU(inplace=True)
        # self.clippedrelu = ReLU(inplace=True)
        self.identity = Conv2d(3,27,(1,1),(1,1),(0,0))
        self.hxblock = HXBlock()
        self.pixelshuffle = PixelShuffle(3)
    def forward(self, LR: torch.Tensor) -> torch.Tensor:
        x = self.cnn_1(LR)
        x = self.relu(x)
        for _ in range(self.m):
            x = self.hxblock(x)
        x = self.cnn_2(x)
        x = self.relu(x)
        x = self.cnn_3(x)
        i = self.identity(LR)
        x += i
        x = self.pixelshuffle(x)
        x[x > 1] = 1
        return x
    
    
    