# Copyright (c) Meta Platforms, Inc. and affiliates.
python run_with_submitit_maskcut_array.py \
--ngpus 1 \
--nodes 1 \
--timeout 1200 \
--partition learnfair \
--vit-arch base \
--patch-size 8 \
--dataset-path /path/to/imagenet/ \
--tau 0.15 \
--out-dir /path/to/save/annotations/ \
--num-folder 2 \
--job-index 0 \
--fixed_size 480 \
--N 3 \