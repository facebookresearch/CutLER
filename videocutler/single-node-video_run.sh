#!/bin/bash
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:12392"
SOCKET_NAME=$(ip r | grep default | awk '{print $5}')
export GLOO_SOCKET_IFNAME=ens32

python -u train_net_video.py --num-gpus 8 --num-machines 1 --machine-rank "$SLURM_NODEID" --dist-url "$DIST_URL" "$@"
