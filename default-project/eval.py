import numpy as np
import os
import pprint
import argparse

import torch
from torch.autograd import Variable

import util
from model import *
from trainer import evaluate


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
            "A new one will be created if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        required=True,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )

    return parser.parse_args()


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    nz, eval_size, num_workers = (
        128,
        10000,
        4,
    )

    # Configure models
    if args.im_size == 32:
        net_g = Generator32()
        net_d = Discriminator32()
    elif args.im_size == 64:
        net_g = Generator64()
        net_d = Discriminator64()
    else:
        raise NotImplementedError(f"Unsupported image size '{args.im_size}'.")

    # Loads checkpoint
    state_dict = torch.load(args.ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader, _, _ = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    # Evaluate models with metrics and samples
    os.makedirs(args.out_dir, exist_ok=True)
    FloatTensor = torch.cuda.FloatTensor
    samples_z = Variable(FloatTensor(np.random.normal(0, 1, (36, 128))))
    samples_save_path = os.path.join(args.out_dir, "samples.png")
    metrics = evaluate(net_g, net_d, eval_dataloader, nz, args.device, samples_z, samples_save_path)
    pprint.pprint(metrics)


if __name__ == "__main__":
    eval(parse_args())
