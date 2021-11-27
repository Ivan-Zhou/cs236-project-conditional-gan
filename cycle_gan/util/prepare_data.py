import os
import glob
import random
from shutil import copy, rmtree


DATASETS = {
    "A": {
        "input_dir": "/home/ubuntu/img_align_celeba/"
    },
    "B": {
        "input_dir": "/home/ubuntu/domain-transfer-net/datasets/emoji_data/images/",
    }
}
TRAIN_TEST_SPLIT = 0.8
SEED = 1024
OUT_DIR = "face2emoji_large"


def create_dir(dirpath):
    if os.path.exists(dirpath):
        rmtree(dirpath)
    os.makedirs(dirpath)


def copy_data(imgs, dst_dir):
    create_dir(dst_dir)
    for img in imgs:
        copy(img, dst_dir)
    print(f"Copied {len(imgs)} images to {dst_dir}")


def process_data(name, input_dir):
    imgs = glob.glob(os.path.join(input_dir, "*.jpg"))
    imgs.extend(glob.glob(os.path.join(input_dir, "*.png")))
    n_imgs = len(imgs)
    assert n_imgs > 1
    print(f"Found {n_imgs} images in {input_dir}.")
    random.Random(SEED).shuffle(imgs)
    n_train = int(n_imgs * TRAIN_TEST_SPLIT)

    imgs_test = imgs[n_train:]
    copy_data(
        imgs=imgs[:n_train],
        dst_dir=os.path.join(OUT_DIR, f"train{name}")
    )

    copy_data(
        imgs=imgs[n_train:],
        dst_dir=os.path.join(OUT_DIR, f"test{name}")
    )


create_dir(OUT_DIR)
for name, content in DATASETS.items():
    input_dir = DATASETS[name]["input_dir"]
    process_data(name, input_dir)
