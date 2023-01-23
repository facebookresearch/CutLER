# Copyright (c) Meta Platforms, Inc. and affiliates.
from .coco import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json
from .builtin import (
    register_all_imagenet, 
    register_all_uvo, 
    register_all_coco_ca,
    register_all_coco_semi,
    register_all_lvis,
    register_all_voc,
    register_all_cross_domain,
    register_all_kitti,
    register_all_objects365,
    register_all_openimages,
    )

__all__ = [k for k in globals().keys() if not k.startswith("_")]