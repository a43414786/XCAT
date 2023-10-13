import os
import glob
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor,Resize,InterpolationMode
import PIL.Image as Image

class RGB2YCbCr(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input: Tensor) -> Tensor:
        if(len(input.shape) == 3):
            return input[0] * 0.257 + input[1] * 0.504 + input[2] * 0.098 + (1/16)
        if(len(input.shape) == 4):
            return input[0,0] * 0.257 + input[0,1] * 0.504 + input[0,2] * 0.098 + (1/16)
        
        # return input[0] * 0.299 + input[1] * 0.587 + input[2] * 0.114

class DIV2K(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.data = glob.glob(os.path.join(self.root, "*.png"))
    
    def __getitem__(self, index)->torch.Tensor:
        HR = Image.open(self.data[index]).convert("RGB")
        HR = ToTensor()(HR)
        LR = torch.nn.functional.interpolate(HR.unsqueeze(dim=0), scale_factor=1/3, mode="bicubic").squeeze(dim=0)
        return LR,HR
    def __len__(self):
        return len(self.data)

class Set14(Dataset):
    def __init__(self,LR_root, HR_root):
        super().__init__()
        self.LR_root = LR_root
        self.HR_root = HR_root
        self.LR = glob.glob(os.path.join(self.LR_root, "*.png"))
        self.HR = glob.glob(os.path.join(self.HR_root, "*.png"))
    
    def __getitem__(self, index)->torch.Tensor:
        HR = Image.open(self.HR[index]).convert("RGB")
        HR = RGB2YCbCr()(ToTensor()(HR)).unsqueeze(dim=0)
        
        LR = Image.open(self.LR[index]).convert("RGB")
        LR = RGB2YCbCr()(ToTensor()(LR)).unsqueeze(dim=0)
        return LR,HR
    def __len__(self):
        return len(self.LR)

class Set5(Set14):
    pass

class PSNR(torch.nn.MSELoss):
    def __init__(self,max = 1.) -> None:
        super().__init__()
        self.max = max
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse = super().forward(input, target)
        if(mse == 0):return 100
        return 20 * torch.log10(self.max /  torch.sqrt(mse))



if __name__ == "__main__":
    data = DIV2K(root="dataset/valid")
    # data = Set5(LR_root="dataset/Set5/LR",HR_root="dataset/Set5/HR")
    # data = Set14(LR_root="dataset/Set14/LR",HR_root="dataset/Set14/HR")
    dataset = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    acc = PSNR()
    
    total = []
    for i, (LR,HR) in enumerate(dataset):
        size = HR.size()[2:]
        rst = Resize(size = size, interpolation = InterpolationMode.BICUBIC,antialias=False)(LR)
        total.append(acc(rst,HR).item())
    import numpy as np
    print(np.average(total))