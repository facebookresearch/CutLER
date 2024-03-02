# VideoCutLER: Unsupervised Video Instance Segmentation

VideoCutLER is a simple unsupervised video instance segmentation (UVIS) method. ***We demonstrate that video instance segmentation models can be learned without using any human annotations, without relying on natural videos (ImageNet data alone is sufficient), and even without motion estimations!***    

<p align="center">
  <img src="../docs/demos_videocutler.gif" width=100%>
</p>            

> [**VideoCutLER: Surprisingly Simple Unsupervised Video Instance Segmentation**](https://people.eecs.berkeley.edu/~xdwang/projects/VideoCutLER/videocutler.pdf)            
> [Xudong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Ishan Misra](https://imisra.github.io/), Ziyun Zeng, [Rohit Girdhar](https://rohitgirdhar.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)             
> UC Berkeley; FAIR, Meta AI            
> CVPR 2024            

[[`arxiv`](https://arxiv.org/abs/2308.14710)] [[`PDF`](https://people.eecs.berkeley.edu/~xdwang/projects/VideoCutLER/videocutler.pdf)] [[`bibtex`](#citation)]             


## Installation
See [installation instructions](INSTALL.md).


## Dataset Preparation
See [Preparing Datasets for VideoCutLER](datasets/README.md).


## Method Overview
<p align="center">
  <img src="../videocutler/docs/videocutler_pipeline.png" width=100%>
</p>
VideoCutLER has three main stages: 
1) Firstly, we generate pseudo-masks for multiple objects in an image using MaskCut. 
2) Then, we convert a random pair of images in the minibatch into a video with corresponding pseudo mask trajectories using ImageCut2Video.
3) Finally, we train an unsupervised video instance segmentation model using these mask trajectories.


## Inference Demo for VideoCutLER with Pre-trained Models

We provide `demo_video/demo.py` that is able to demo builtin configs. Run it with:
```
cd videocutler
python demo_video/demo.py \
  --config-file configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml \
  --input docs/demo-videos/99c6b1acf2/*.jpg \
  --confidence-threshold 0.8 \
  --output demos/ \
  --opts MODEL.WEIGHTS videocutler_m2f_rn50.pth
```
Our trained VideoCutLER model on synthetic videos using ImageNet-1K can be obtained from [here](https://drive.google.com/file/d/11TACB8tOaAc-eXBo_i2arR_7qgGSeXRg/view?usp=drive_link). Then you should specify `MODEL.WEIGHTS` to the model checkpoint for evaluation.
Above command will run the inference and show visualizations in an OpenCV window, and save the results in the mp4 format.
For details of the command line arguments, see `demo.py -h` or look at its source code to understand its behavior. Some common arguments are:
<!-- * To save outputs to a directory (for videos), use `--output`. -->
* To get a higher recall, use a smaller `--confidence-threshold`.
* To save each frame's segmentation result, add `--save-frames True` before `--opts`.
* To save each frame's segmentation masks, add `--save-masks True` before `--opts`.

Following, we give some visualizations of the model predictions on the demo videos.
<p align="center">
  <img src="docs/videocutler_demos.gif" width=100%>
</p>


### Unsupervised Model Learning
We provide a script `train_net_video.py`, that is made to train all the configs provided in VideoCutLER.
To train a model with "train_net_video.py", first setup the ImageNet-1K dataset following [datasets/README.md](../datasets/README.md).

Before training the detector, it is necessary to use MaskCut to generate pseudo-masks for all ImageNet data.
You can either use the pre-generated json file directly by downloading it from [here]() and placing it under "DETECTRON2_DATASETS/imagenet/annotations/", or generate your own pseudo-masks by following the instructions in [MaskCut](#1-maskcut).
You should download the pre-trained CutLER model from this [link](https://drive.google.com/file/d/1YFP14mCHBGR3SbepGiTv-OMeUIrtTPc3/view?usp=sharing) and then place it in the "videocutler/pretrain" directory, then run:
```
cd videocutler
export DETECTRON2_DATASETS=/path/to/DETECTRON2_DATASETS/
python train_net_video.py \
  --config-file configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml \
  SOLVER.BASE_LR 0.000005 SOLVER.IMS_PER_BATCH 16 MODEL.MASK_FORMER.DROPOUT 0.3 \
  OUTPUT_DIR OUTPUT-DIR/ \
```
For more options, see `python train_net_video.py -h`.

If you want to train a model using multiple nodes, you may need to adjust [some model parameters](https://arxiv.org/abs/1706.02677) and some SBATCH command options in "train-1node.sh" and "single-node-video_run.sh", then run:
```
cd videocutler
export DETECTRON2_DATASETS=/path/to/DETECTRON2_DATASETS/
sbatch train-1node.sh \
  --config-file configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml \
  SOLVER.BASE_LR 0.000005 SOLVER.IMS_PER_BATCH 16 MODEL.MASK_FORMER.DROPOUT 0.3 \
  OUTPUT_DIR OUTPUT-DIR/ \
```


### Unsupervised Zero-shot Evaluation
To evaluate a model's performance on YouTubeVIS-2019 and YouTubeVIS-2021, please refer to [datasets/README.md](datasets/README.md) for instructions on preparing the datasets. Next, download the [model weights](https://drive.google.com/file/d/11TACB8tOaAc-eXBo_i2arR_7qgGSeXRg/view?usp=drive_link), specify the "model_weights", "config_file" and the path to "DETECTRON2_DATASETS", then run the following commands. 
```
export DETECTRON2_DATASETS=/PATH/TO/DETECTRON2_DATASETS/
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_video.py --num-gpus 4 \
  --config-file configs/imagenet_video/videocutler_eval_ytvis2019.yaml \
  --eval-only MODEL.WEIGHTS videocutler_m2f_rn50.pth \
  OUTPUT_DIR OUTPUT-DIR/ytvis_2019

python eval_ytvis.py --dataset-path ${DETECTRON2_DATASETS} --dataset-name 'ytvis_2019' --result-path 'OUTPUT-DIR/ytvis_2019/'
```


## Ethical Considerations
VideoCutLER's wide range of video instance segmentation capabilities may introduce similar challenges to many other visual recognition methods. As the video can contain arbitrary instances, it may impact the model output.


## How to get support from us?
If you have any general questions, feel free to email us at [Xudong Wang](mailto:xdwang@eecs.berkeley.edu). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{wang2023videocutler,
  title={VideoCutLER: Surprisingly Simple Unsupervised Video Instance Segmentation},
  author={Wang, Xudong and Misra, Ishan and Zeng, Ziyun and Girdhar, Rohit and Darrell, Trevor},
  journal={arXiv preprint arXiv:2308.14710},
  year={2023}
}
```
