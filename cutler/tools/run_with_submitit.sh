# Copyright (c) Meta Platforms, Inc. and affiliates.
sbatch tools/train-1node.sh \
  --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
  OUTPUT_DIR /path/to/output