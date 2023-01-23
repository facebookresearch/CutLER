# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/__init__.py

from fvcore.transforms.transform import *
from .transform import *
from detectron2.data.transforms.augmentation import *
from .augmentation_impl import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata