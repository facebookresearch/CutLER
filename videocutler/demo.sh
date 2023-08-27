python demo_video/demo.py \
  --config-file configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml \
  --input docs/demo-videos/99c6b1acf2/*.jpg \
  --confidence-threshold 0.8 \
  --output demos/ \
  # --save-frames True \
  # --save-masks True \
  --opts MODEL.WEIGHTS videocutler_m2f_rn50.pth