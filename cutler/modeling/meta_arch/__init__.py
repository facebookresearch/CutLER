# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/__init__.py

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

__all__ = list(globals().keys())
