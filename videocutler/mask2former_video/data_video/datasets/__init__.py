# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/Mask2Former/tree/main/mask2former_video

from . import builtin  # ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
