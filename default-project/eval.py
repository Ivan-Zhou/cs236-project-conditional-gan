import glob
import json
import numpy as np
import os
import pprint
import argparse

import torch
from torch.autograd import Variable

import util
from model import *
from trainer import evaluate, evaluate_v2


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
        "--dataset",
        type=str,
        default="mnist",
        help="mnist|stanford_dog",
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
        default="",
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="",
        help="Directory to the experiment folder that contains the checkpoints to load.",
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
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="default|cgan",
    )
    return parser.parse_args()


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """
    pprint.pprint(vars(args))

    # Set parameters
    nz, eval_size, num_workers = (
        128,
        10000,
        4,
    )

    num_classes = util.get_num_classes_by_dataset(args.dataset)

    # Configure models
    if args.model == "cgan" or args.model == "lscgan":
        net_g = CGANGenerator(nz, (3, args.im_size, args.im_size), num_classes = num_classes)
        net_d = CGANDiscriminator((3, args.im_size, args.im_size), num_classes = num_classes)
    elif args.im_size == 32:
        net_g = Generator32()
        net_d = Discriminator32()
    elif args.im_size == 64:
        net_g = Generator64()
        net_d = Discriminator64()
    else:
        raise NotImplementedError(f"Unsupported model type {args.model}.")

    # Loads checkpoint
    ckpt_path = util.get_eval_checkpoint(args.ckpt_path, args.exp_dir)
    state_dict = torch.load(ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader, _, _ = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers, dataset=args.dataset
    )

    # Evaluate models with metrics and samples
    os.makedirs(args.out_dir, exist_ok=True)
    FloatTensor = torch.cuda.FloatTensor
    samples_save_path = os.path.join(args.out_dir, "samples.png")
    samples_z = Variable(FloatTensor(np.random.normal(0, 1, (36, nz))))
    if args.model == "cgan":
        metrics, _ = evaluate_v2(net_g, net_d, eval_dataloader, nz, num_classes, args.device, samples_z, samples_save_path, "hinge")
    elif args.model == "lscgan":
        metrics, _ = evaluate_v2(net_g, net_d, eval_dataloader, nz, num_classes, args.device, samples_z, samples_save_path, "mse")
    elif args.model == "default":
        metrics, _ = evaluate(net_g, net_d, eval_dataloader, nz, args.device, samples_z, samples_save_path)
    else:
        raise NotImplementedError(f"Unsupported model type {args.model}.") 
    pprint.pprint(metrics)
    metrics_save_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_save_path, "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    eval(parse_args())
