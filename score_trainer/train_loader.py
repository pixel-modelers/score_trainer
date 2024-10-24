from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import glob
import h5py
import numpy as np


class Dset(Dataset):

    def __init__(self, glob_s, verbose=False):
        self.fnames = glob.glob(glob_s)
        self.h5s = []  # list of file handle
        self.h5_idx = [] # list of file_handle_idx, h5_dset_idx
        self.num_tot = 0
        self.verbose = verbose

        self._open_fnames()

    def __len__(self):
        return self.num_tot

    def __getitem__(self, item):
        if item <0 or item >=self.num_tot:
            raise IndexError(f"Index out of bounds, should be >=0 and < {self.num_tot}")
        h5_idx, h5_dset_idx = self.h5_idx[item]
        h5 = self.h5s[h5_idx]
        img = h5['images'][h5_dset_idx]
        lab = h5['labels'][h5_dset_idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        return torch.tensor(img), torch.tensor(lab[None])

    def _open_fnames(self):
        for i_f, f in enumerate(self.fnames):
            h = h5py.File(f, 'r')
            self.h5s.append(h)
            nlab = h['labels'].shape[0]
            if self.verbose:
                print("Found %d training examples in %s (%d/%d)" %(nlab, f, i_f+1, len(self.fnames)))
            self.num_tot += nlab
            self.h5_idx += list(zip([i_f]*nlab, range(nlab)))
        print("Done. Found %d examples total" % self.num_tot)


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("glob",type=str, help="glob of input score_trainer training data files to load")
    args = ap.parse_args()
    import sys
    ds = Dset(args.glob, verbose=True)

    import numpy as np
    import pylab as plt
    inds = np.random.permutation(len(ds))
    f, ax = plt.subplots(1, 2)
    for a in ax:
        a.imshow(np.random.random((10, 10)), cmap="gray_r")
        a.grid(1)
    for i in inds:
        img, lab = ds[i]
        lab = int(lab)
        cl = img[0].min(), img[0].max()
        for a in ax:
            a.images[0].set_clim(cl)
        ax[0].images[0].set_data(img[0])
        ax[1].images[0].set_data(img[1])
        s = "subimg %d is " %i
        s += "Good!" if lab == 1 else "Bad :("
        plt.suptitle(s)
        plt.draw()
        plt.waitforbuttonpress()
