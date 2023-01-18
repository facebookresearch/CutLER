# Copyright (c) Meta Platforms, Inc. and affiliates.

from .boxes import pairwise_iou_max_scores

__all__ = [k for k in globals().keys() if not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
