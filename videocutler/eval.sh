export DETECTRON2_DATASETS=/shared/xudongw/DATASETS/

###### eval YouTubeVIS-2019 ######
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_video.py --num-gpus 4 \
  --config-file configs/imagenet_video/videocutler_eval_ytvis2019.yaml \
  --eval-only MODEL.WEIGHTS videocutler_m2f_rn50.pth \
  OUTPUT_DIR OUTPUT/ytvis_2019

python eval_ytvis.py --dataset-path ${DETECTRON2_DATASETS} --dataset-name 'ytvis_2019' --result-path 'OUTPUT/ytvis_2019/'

###### eval YouTubeVIS-2021 ######
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_video.py --num-gpus 4 \
#   --config-file configs/imagenet_video/videocutler_eval_ytvis2021.yaml \
#   --eval-only MODEL.WEIGHTS videocutler_m2f_rn50.pth \
#   OUTPUT_DIR OUTPUT/ytvis_2021/

# python eval_ytvis.py --dataset-path ${DETECTRON2_DATASETS} --dataset-name 'ytvis_2021' --result-path 'OUTPUT/ytvis_2021/'