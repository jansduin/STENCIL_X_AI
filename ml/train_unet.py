import os, glob, random, math
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# --- Simple U-Net (small) ---
class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2); self.p2=nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4); self.p3=nn.MaxPool2d(2)
        self.b  = DoubleConv(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        self.act = nn.Sigmoid()
    def forward(self,x):
        d1=self.d1(x); d2=self.d2(self.p1(d1)); d3=self.d3(self.p2(d2))
        b=self.b(self.p3(d3))
        u3=self.u3(b); c3=self.c3(torch.cat([u3,d3],1))
        u2=self.u2(c3); c2=self.c2(torch.cat([u2,d2],1))
        u1=self.u1(c2); c1=self.c1(torch.cat([u1,d1],1))
        y=self.act(self.out(c1))
        return y

# --- Dataset of paired images ---
class PairDS(Dataset):
    def __init__(self, root='dataset', size=512):
        self.A = sorted(glob.glob(os.path.join(root,'originals','*')))
        self.B = sorted(glob.glob(os.path.join(root,'stencils','*')))
        assert len(self.A)==len(self.B) and len(self.A)>0, 'Need paired dataset'
        self.size=size
        self.tf = T.Compose([T.Grayscale(num_output_channels=1), T.Resize((size,size)), T.ToTensor()])
    def __len__(self): return len(self.A)
    def __getitem__(self,i):
        a = Image.open(self.A[i]).convert('RGB')
        b = Image.open(self.B[i]).convert('RGB')
        a = self.tf(a)  # [1,H,W] 0..1
        b = self.tf(b)
        return a, b

def train(root='dataset', epochs=30, bs=4, lr=1e-3, out='stencil_model_fp32.pth'):
    ds = PairDS(root); n=len(ds); ntr=int(n*0.9)
    tr, va = torch.utils.data.random_split(ds, [ntr, n-ntr])
    dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2)
    dv = DataLoader(va, batch_size=1, shuffle=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = UNetSmall().to(dev)
    l1 = nn.L1Loss()
    def ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
        mu_x = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
        mu_y = torch.nn.functional.avg_pool2d(y, 3, 1, 1)
        sigma_x = torch.nn.functional.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
        sigma_y = torch.nn.functional.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
        sigma_xy = torch.nn.functional.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
        return 1 - ssim.mean()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = 1e9
    for ep in range(1,epochs+1):
        net.train(); tl=0
        for a,b in dl:
            a,b = a.to(dev), b.to(dev)
            y = net(a)
            loss = l1(y,b) + 0.2*ssim_loss(y,b)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*a.size(0)
        net.eval(); vl=0
        with torch.no_grad():
            for a,b in dv:
                a,b = a.to(dev), b.to(dev)
                y = net(a)
                loss = l1(y,b) + 0.2*ssim_loss(y,b)
                vl += loss.item()
        tl /= len(dl.dataset); vl /= len(dv.dataset)
        print(f"epoch {ep:02d} train {tl:.4f} val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(net.state_dict(), out)
            print("  saved", out)

if __name__ == '__main__':
    train()
