from __future__ import division

import torch, torchvision
import numpy as np
import os

from score_trainer.train import LeNet


class roiCheck:
    def __init__(self, state_file=None):
        if state_file is None:
            state_file = os.path.join(os.path.dirname(__file__), "state_ep10.net")
            if not os.path.exists(state_file):
                raise OSError(f"Default state file {state_file} does not exists, please run `score.getMod`")
        self.state_file = state_file
        state = torch.load(self.state_file, weights_only=True)
        self.model = LeNet()
        self.model.load_state_dict(state)
        self.model.eval()
        self.model = self.model.to("cpu")
        self.rs = torchvision.transforms.Resize((32,32), antialias=True)
        self.sig = torch.nn.Sigmoid()
   
    def score(self, dat_im, mod_im): 
        assert type(dat_im)==type(mod_im)
        if isinstance(dat_im, list):
            assert isinstance(dat_im[0], np.ndarray)
            assert len(dat_im[0].shape)==2
            T = torch.tensor(np.array(list(zip(dat_im, mod_im))).astype(np.float32))
        else:
            assert isinstance(dat_im, np.ndarray)
            assert len(dat_im.shape)==2
            T = torch.tensor(np.array([dat_im,mod_im])[None].astype(np.float32))

        T = self.rs(T)
        pred = self.sig(self.model(T))
        pred = [p.item() for p in pred]
        if isinstance(dat_im, np.ndarray):
            assert len(pred)==1
            pred = pred[0]

        return pred
