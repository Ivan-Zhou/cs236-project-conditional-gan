import os
import pprint
import argparse

import torch

import torchvision.utils as vutils

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  

import util
from model import *
from trainer import evaluate

nz, eval_size, num_workers = (
        128,
        10000,
        4,
    )
data_dir = "./data/Images/"
im_size = 32
batch_size = 64
ckpt_path = "./out/test/ckpt/150000.pth"
net_g = Generator32()
net_d = Discriminator32()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
state_dict = torch.load(ckpt_path)
net_g.load_state_dict(state_dict["net_g"])
net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
_, eval_dataloader, _, _ = util.get_dataloaders(
data_dir, im_size, batch_size, eval_size, num_workers
)

# Evaluate models
metrics = evaluate(net_g, net_d, eval_dataloader, nz, device)
# -

pprint.pprint(metrics)

samples_z = torch.randn((36, nz), device=device)

samples = net_g(samples_z)
samples = F.interpolate(samples, 256).cpu()
samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)

samples.shape

samples

samples.permute(1,2,0).shape

samples.permute(1,2,0).detach().cpu().numpy().shape

plt.rcParams['figure.figsize'] = [40, 20]

plt.imshow(samples.permute(1,2,0).detach().cpu().numpy())


