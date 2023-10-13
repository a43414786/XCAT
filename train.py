import torch
from torch.autograd import Variable
import torch.nn as nn
from XCAT import *
from dataset import *

import numpy as np

M_NUM = 8

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self,eps = 1e-1):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class Train():
    def __init__(self,is_cuda= False,m = 12) -> None:
        self.is_cuda = is_cuda
        self.m = m
        self.model = XCAT(m)
        if(is_cuda):
            self.model = self.model.cuda()
        self.criterian = self.getCriterian()
        data = DIV2K(root="dataset/train")
        self.train_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        data = DIV2K(root="dataset/valid")
        self.val_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        
    def getCriterian(self):
        return L1_Charbonnier_loss(eps=0.1)
    
    def get_lr(self,epoch):
        if(epoch == 0):
            return 1e-3
        if(epoch == 1):
            return 1.375e-3
        if(epoch == 2):
            return 1.75e-3
        if(epoch == 3):
            return 2.125e-3
        if(epoch == 4):
            return 2.5e-3
        else:
            return 2.5e-3 - 2.4e-3 * ((epoch - 4) / 45)
    
    def getOptimizer(self,epoch):
        return torch.optim.Adam(params = self.model.parameters(),
                                          lr = self.get_lr(epoch),
                                          betas=(0.9, 0.999),
                                          eps=1e-8)
    
    def val(self):
        total = []
        for idx,(LR,HR) in enumerate(self.val_loader):
            LR = Variable(LR)
            HR = Variable(HR)
            if(self.is_cuda):
                LR = LR.cuda()
                HR = HR.cuda()
            out = self.model(LR)
            total.append(PSNR()(RGB2YCbCr()(out),RGB2YCbCr()(HR)).item())
            print(idx,end = '\r')
        return np.average(total)    
    
    def save(self,index,train_psnr,val_psnr):
        if(not os.path.exists(f'{self.m}')):
            os.mkdir(f'{self.m}')
    
        if(index == 0):
            with open(f'{self.m}/log.csv','w') as f:
                f.writelines('epoch,train_psnr,val_psnr\n')
        with open(f'{self.m}/log.csv','a') as f:
            f.writelines(f'{index},{train_psnr},{val_psnr}\n')
        if(index % 10 == 9):
            torch.save(self.model.state_dict(),f"{self.m}/{index}.pth")
            
    def train(self):
        for i in range(50):
            optimizer = self.getOptimizer(i)
            total = []
            for idx,(LR,HR) in enumerate(self.train_loader):
                LR = Variable(LR)
                HR = Variable(HR)
                if(self.is_cuda):
                    LR = LR.cuda()
                    HR = HR.cuda()
                out = self.model(LR)
                loss = self.criterian(out,HR)
                total.append(PSNR()(RGB2YCbCr()(out),RGB2YCbCr()(HR)).item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f' {idx}',end = '\r')
            train_psnr = np.average(total)                
            val_psnr = self.val()                
            self.save(i,train_psnr,val_psnr)
            print(f"epoch:{i} PSNR:{train_psnr} val_PSNR:{val_psnr}")

if __name__ == '__main__':
    train = Train(is_cuda=True,m=M_NUM)
    train.train()
