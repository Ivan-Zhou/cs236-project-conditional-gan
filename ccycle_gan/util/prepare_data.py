import os
import glob
import random
from shutil import copy, rmtree
import pandas as pd


DATASETS = {
    "A": {
        "input_dir": "/home/ubuntu/img_align_celeba/",
        "attribute_file_path": "util/celeba_attributes.csv",
        "target_attribute": "Eyeglasses",
        "target_attribute_type": int,
        "name_field": "Name",
    },
    "B": {
        "input_dir": "/home/ubuntu/domain-transfer-net/datasets/emoji_data/images/",
        "attribute_file_path": "util/emoji_attributes.csv",
        "target_attribute": "glasses",
        "target_attribute_type": str,
        "name_field": "image_name",
    },
}
TRAIN_TEST_SPLIT = 0.8
SEED = 1024
MAX_EXAMPLES = 5000
OUT_DIR = "face2emoji_conditional_5k"


def create_dir(dirpath, exist_ok=False):
    if not exist_ok and os.path.exists(dirpath):
        rmtree(dirpath)
    os.makedirs(dirpath, exist_ok=exist_ok)


def copy_img(data):
    img_path = data["IMAGE_PATH"]
    has_glass = data["has_glass"]
    folder = "has_glass" if has_glass else "no_glass"
    dst_dir = os.path.join(data["dst_par_dir"], folder)
    create_dir(dst_dir, exist_ok=True)
    copy(img_path, dst_dir)


def copy_data(imgs, df_attributes, dst_dir):
    create_dir(dst_dir)
    df_imgs = pd.DataFrame({"IMAGE_PATH": imgs})
    df_imgs["image_name"] = df_imgs["IMAGE_PATH"].apply(lambda x: os.path.basename(x))
    df_attributes_subset = pd.merge(
        left=df_imgs,
        right=df_attributes,
        how="inner",
        on="image_name",
    )
    assert len(df_attributes_subset) == len(imgs), print(
        len(df_attributes_subset), len(imgs)
    )
    df_attributes_subset["dst_par_dir"] = dst_dir
    df_attributes_subset.apply(copy_img, axis=1)
    print(f"Copied {len(df_attributes_subset)} images to {dst_dir}")


def read_attribute(attribute_path, target_attribute, target_attribute_type, name_field):
    df_attributes = pd.read_csv(metadata["attribute_file_path"]).fillna("")
    if target_attribute_type == str:
        df_attributes["has_glass"] = df_attributes[target_attribute] != ""
    elif target_attribute_type == int:
        df_attributes["has_glass"] = df_attributes[target_attribute] > 0
    else:
        raise ValueError(f"Attribute type {target_attribute_type} not supported!")
    df_attributes["image_name"] = df_attributes[name_field]
    return df_attributes


def process_data(name, metadata):
    input_dir = metadata["input_dir"]
    imgs = glob.glob(os.path.join(input_dir, "*.jpg"))
    imgs.extend(glob.glob(os.path.join(input_dir, "*.png")))
    n_imgs = len(imgs)
    assert n_imgs > 1
    print(f"Found {n_imgs} images in {input_dir}.")
    random.Random(SEED).shuffle(imgs)
    if MAX_EXAMPLES is not None:
        n_imgs = MAX_EXAMPLES
        print(f"Select {n_imgs} example images")
        imgs = imgs[:n_imgs]
    n_train = int(n_imgs * TRAIN_TEST_SPLIT)

    df_attributes = read_attribute(
        metadata["attribute_file_path"],
        metadata["target_attribute"],
        metadata["target_attribute_type"],
        metadata["name_field"],
    )

    copy_data(
        imgs=imgs[:n_train],
        dst_dir=os.path.join(OUT_DIR, f"train{name}"),
        df_attributes=df_attributes,
    )

    copy_data(
        imgs=imgs[n_train:],
        dst_dir=os.path.join(OUT_DIR, f"test{name}"),
        df_attributes=df_attributes,
    )


create_dir(OUT_DIR)
for name, metadata in DATASETS.items():
    process_data(name, metadata)
