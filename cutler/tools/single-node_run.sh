# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/bin/bash
export DETECTRON2_DATASETS=/path/to/DETECTRON2_DATASETS/
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:12399"
SOCKET_NAME=$(ip r | grep default | awk '{print $5}')
export GLOO_SOCKET_IFNAME=$SOCKET_NAME

python -u train_net.py --num-gpus 8 --num-machines 1 --machine-rank "$SLURM_NODEID" --dist-url "$DIST_URL" "$@"