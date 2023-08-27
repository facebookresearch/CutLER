# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/Mask2Former/tree/main/mask2former_video

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_imagenet_cls_agnostic_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_train_5perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_5percent.json"),
    "ytvis_2019_train_10perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_10percent.json"),
    "ytvis_2019_train_20perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_20percent.json"),
    "ytvis_2019_train_30perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_30percent.json"),
    "ytvis_2019_train_40perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_40percent.json"),
    "ytvis_2019_train_50perc": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_50percent.json"),
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_minus_2019_train": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_val_sub.json"),
}

_PREDEFINED_SPLITS_ImageNet_CLS_AGNOSTIC = {
    "imagenet_video_train_cls_agnostic": ("imagenet/train",
                        "imagenet/annotations/video_imagenet_train_fixsize480_tau0.15_N3.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_imagenet_cls_agnostic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ImageNet_CLS_AGNOSTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_imagenet_cls_agnostic_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_imagenet_cls_agnostic(_root)