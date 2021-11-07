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
from trainer import evaluate, evaluate_v2

nz, eval_size, num_workers = (
        128,
        10000,
        4,
    )
data_dir = "./data/Images/"
im_size = 32
batch_size = 64
#num_classes = 120
num_classes = 10
ckpt_path = "./out/debug_cgan_mnist/ckpt/15000.pth"
num_channels=3
net_g = CGANGenerator(nz, (num_channels, im_size, im_size), num_classes)
net_d = CGANDiscriminator((num_channels, im_size, im_size), num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
state_dict = torch.load(ckpt_path)
net_g.load_state_dict(state_dict["net_g"])
net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
_, eval_dataloader, _, _ = util.get_dataloaders(
data_dir, im_size, batch_size, eval_size, num_workers, dataset="mnist"
)

# Evaluate models
metrics = evaluate_v2(net_g, net_d, eval_dataloader, nz,num_classes,device)
# -

pprint.pprint(metrics)



samples_z = torch.randn((36, nz), device=device)
#gen_labels = Variable(torch.LongTensor(np.arange(samples_z.shape[0]))).to(device)
gen_labels = Variable(torch.LongTensor(np.ones(samples_z.shape[0]))).to(device)
gen_labels = Variable(torch.LongTensor(np.ones(samples_z.shape[0]))*6).to(device)

gen_labels

samples = net_g(samples_z, gen_labels)
samples = F.interpolate(samples, 256).cpu()
samples = vutils.make_grid(samples, nrow=6, padding=5, normalize=True)

samples.permute(1,2,0).shape

samples.permute(1,2,0).detach().cpu().numpy().shape

plt.rcParams['figure.figsize'] = [40, 20]

plt.imshow(samples.permute(1,2,0).detach().cpu().numpy())


