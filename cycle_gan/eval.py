import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
import wandb
import numpy as np
from PIL import Image
from torchmetrics import IS, FID, KID
import torch
import json
from pathlib import Path


def visualize(dataset, model, opt, n_row=8, n_col=4, im_size=256):
    num_test = n_row * n_col
    output = np.zeros((n_row * im_size, n_col * 2 * im_size, 3), np.uint8)
    for i, data in enumerate(dataset):
        if i >= num_test:
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        image_pair = np.zeros((im_size, 2 * im_size, 3), np.uint8)
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data)
            if label == "real_A":  # real image
                image_pair[:, :im_size] = im
            elif label == "fake_B":  # generated emoji
                image_pair[:, im_size:] = im

        row_idx = int(i / n_col)
        col_idx = i % n_col
        output[
            row_idx * im_size : (row_idx + 1) * im_size,
            col_idx * 2 * im_size : (col_idx + 1) * 2 * im_size,
        ] = image_pair
    output_path = os.path.join(
        opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}", f"{n_row}_{n_col}.png"
    )
    im = Image.fromarray(output)
    im.save(output_path)
    print(f"Sample images are saved into {output_path}.")


def compute_metrics(dataset, model, opt, device="cuda", num_test=200):
    num_test = min(num_test, len(dataset))

    is_, fid, kid = (
        IS().to(device),
        FID().to(device),
        KID(subset_size=int(num_test / 2)).to(device),
    )
    for i, data in enumerate(dataset):
        if i > num_test:
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        real, fake = None, None
        for label, im_data in visuals.items():
            if label == "fake_B":  # generated emoji
                fake = im_data.to(torch.uint8)
            elif label == "real_B":  # real emoji
                real = im_data.to(torch.uint8)
        assert real is not None
        assert fake is not None
        is_.update(fake)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        kid.update(real, real=True)
        kid.update(fake, real=False)

        if i % 50 == 0:  # save images to an HTML file
            print("Evaluate (%04d)-th image out of %s" % (i, num_test))

    metrics = {
        "IS": is_.compute()[0].item(),
        "FID": fid.compute().item(),
        "KID": kid.compute()[0].item(),
    }
    print(metrics)
    metrics_save_path = os.path.join(
        opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}", f"metrics.json"
    )
    with open(metrics_save_path, "w") as outfile:
        json.dump(metrics, outfile)
    print(f"Metrics are saved into {metrics_save_path}.")


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Create output_dir and the parent dir
    output_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    visualize(dataset, model, opt)
    compute_metrics(dataset, model, opt)
