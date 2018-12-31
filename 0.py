import torch
from torch import nn
import torch.nn.functional as F
import os

# disable x
import matplotlib

matplotlib.use("GTK3Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from config import o
from data import BSD, BSD3000, Levin, Sun
from util import center_crop, change_key, crop, isnan, load, log, mean, npsnr, npsnr_align_max, show


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        filter_size = 5
        filter_num = 24
        penalty_num = 63
        self.mean = torch.linspace(-310, 310, penalty_num).view(1, 1, penalty_num, 1, 1)
        self.weight = torch.randn(1, filter_num, penalty_num, 1, 1) * 0.1
        self.filter = torch.randn(filter_num, 1, filter_size, filter_size) * 0.1
        self.pad = nn.ReplicationPad2d(filter_size // 2)
        self.crop = nn.ReplicationPad2d(-(filter_size // 2))
        self.weight = nn.Parameter(self.weight)
        self.filter = nn.Parameter(self.filter)

    # Bx1xHxW
    def forward(self, x):
        xx = x
        self.mean = self.mean.to(x.device)
        x = self.pad(x)
        x = F.conv2d(x, self.filter)
        x = x.unsqueeze(2)
        x = ((x - self.mean).pow(2) / -200).exp() * self.weight
        x = x.sum(2)
        # todo? conv_transpose2d vs rotate 180
        x = F.conv_transpose2d(x, self.filter)
        x = self.crop(x)
        # x = self.pad(x)
        # x = F.conv2d(x, self.filter.permute(1,0,2,3).flip(2, 3))
        # log("filter", self.filter)
        return xx - x


o.device = "cuda" if torch.cuda.is_available() else "cpu"
print("use " + o.device)

# m:model to train, p:pre models
def train(m, p=None):
    d = DataLoader(BSD3000(), o.batch_size, num_workers=o.num_workers)
    optimizer = torch.optim.Adam(m.parameters(), lr=o.lr)
    iter_num = len(d)
    num = 0
    losss = []
    stage = 1 if not p else p.stage + 1
    for epoch in range(o.epoch):
        for i in tqdm(d):
            g, y, k, s = [x.to(o.device) for x in i]
            x = y
            optimizer.zero_grad()
            out = m(x)
            log("out", out)
            loss = npsnr(out, g)
            loss.backward()
            optimizer.step()
            losss.append(loss.detach().item())
            assert not isnan(losss[-1])
            print("stage", stage, "epoch", epoch + 1)
            log("loss", mean(losss[-5:]))
            num += 1
            # if num > (o.epoch * iter_num - 4):
            if num % 50 == 1:
                show(
                    torch.cat((y[0, 0], g[0, 0], out[0, 0]), 1),
                    # save=f"save/{stage:02}{epoch:02}.png",
                )
    plt.clf()
    plt.plot(range(len(losss)), losss)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title(f"{iter_num} iter x {o.epoch} epoch")
    plt.savefig(f"save/{stage:02}loss.png")


m = DataParallel(M()).to(o.device)
train(m)
