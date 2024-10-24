from __future__ import print_function, division
from argparse import ArgumentParser

def get_args():
    pa = ArgumentParser()
    pa.add_argument("glob", type=str)
    pa.add_argument("odir", type=str)
    pa.add_argument("--bs", type=int, default=512, help="batch size")
    pa.add_argument("--lr", type=float, default=1e-3, help="learing rate")
    pa.add_argument("--gpuid", default=None, type=int, help="GPU Id")
    pa.add_argument("--nepoch", type=int, default=10, help="number of epochs (passes through training data)")
    args = pa.parse_args()
    return args

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from score_trainer.train_loader import Dset
import os
from torchvision.transforms import Resize


class LeNet(nn.Module):
    def __init__(self, dev=None):
        """
        :param dev: pytorch device
        """
        super().__init__()
        if dev is None:
            self.dev = "cuda:0"
        else:
            self.dev = dev
        self.conv1 = nn.Conv2d(2, 6, 3, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev)

        self.fc1 = nn.Linear(128, 80, device=self.dev)
        self.fc2 = nn.Linear(80, 1, device=self.dev)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    args = get_args()
    ds = Dset(args.glob)

    ntot = len(ds)
    nval = ntest = int(.2*ntot)

    train_ds, val_ds, test_ds = random_split(ds, lengths=[ntot-2*nval, nval, ntest])
    train_dl = DataLoader(train_ds,batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(val_ds,batch_size=args.bs)
    test_dl = DataLoader(test_ds, batch_size=args.bs)

    dev = "cpu"
    if args.gpuid is not None:
        dev = "cuda:%d" % args.gpuid
    model = LeNet(dev=dev)
    optim = Adam(model.parameters(), lr=args.lr)
    calc_loss = torch.nn.BCELoss()

    rs = Resize((32,32), antialias=True)
    sig = torch.nn.Sigmoid()

    for i_ep in range(args.nepoch):
        model.train()
        for i_batch, (imgs, labs) in enumerate(train_dl):
            optim.zero_grad()
            imgs = rs(imgs)
            imgs = imgs.to(dev)
            labs = labs.to(dev)
            preds = model(imgs)
            preds = sig(preds)
            loss = calc_loss(preds, labs)
            loss.backward()
            optim.step()
            print(f"Ep{i_ep+1}; batch {i_batch+1}/{len(train_dl)}, loss={loss.item():.5f}", end="\r", flush=True)
        print(f"\nDone with Epoch {i_ep+1}/{args.nepoch}")
        model.eval()
        with torch.no_grad():
            val_nmatch = 0
            for i, (imgs, labs) in enumerate(val_dl):
                imgs = rs(imgs)
                imgs = imgs.to(dev)
                labs = labs.to(dev)
                preds = model(imgs)
                preds = torch.round(sig(preds))
                val_nmatch += torch.sum(preds==labs).item()
                print(f"val batch {i+1}/{len(test_dl)}", end="\r",flush=True )
            print("")
            test_nmatch = 0
            for i, (imgs, labs) in enumerate(test_dl):
                imgs = rs(imgs)
                imgs = imgs.to(dev)
                labs = labs.to(dev)
                preds = model(imgs)
                preds = torch.round(sig(preds))
                test_nmatch += torch.sum(preds==labs).item()
                print(f"test batch {i+1}/{len(test_dl)}", end="\r", flush=True)
            print("")
            val_acc = val_nmatch / len(val_ds)
            test_acc = test_nmatch / len(test_ds)
            print(f"Acc at Ep {i_ep+1} ; valAcc: {val_acc:.4f} , testAcc: {test_acc:.4f} ")
            # save the model weights
            model_file = os.path.join(args.odir, f"state_ep{i_ep+1}.net")
            if not os.path.exists(args.odir):
                os.makedirs(args.odir)
            torch.save(model.state_dict(), model_file)
