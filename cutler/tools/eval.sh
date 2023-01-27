# Copyright (c) Meta Platforms, Inc. and affiliates.

# link to the dataset folder, model weights and the config file.
export DETECTRON2_DATASETS=/path/to/DETECTRON2_DATASETS/
model_weights="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth"
config_file="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"
num_gpus=2

echo "========== start evaluating the model on all 11 datasets =========="

test_dataset='cls_agnostic_clipart'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_watercolor'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_comic'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_voc'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_objects365'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_openimages'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_kitti'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_coco'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_coco20k'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only MODEL.WEIGHTS ${model_weights}

test_dataset='cls_agnostic_lvis'
echo "========== evaluating ${test_dataset} =========="
# LVIS should set TEST.DETECTIONS_PER_IMAGE=300
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only MODEL.WEIGHTS ${model_weights} TEST.DETECTIONS_PER_IMAGE 300

test_dataset='cls_agnostic_uvo'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only MODEL.WEIGHTS ${model_weights}

echo "========== evaluation is completed =========="