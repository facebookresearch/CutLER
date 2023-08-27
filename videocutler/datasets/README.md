# Prepare Datasets for VideoCutLER

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

VideoCutLER has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  imagenet/
  ytvis_2019/
  ytvis_2021/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

Please check expected dataset structure for ImageNet-1K at [here](../../datasets/README.md). You can directly [download](https://drive.google.com/file/d/1gllHvrZQNVXphnk-IQxMcXh87Qs86ofT/view?usp=sharing) the pre-processed ImageNet-1K annotations produced by MaskCut in YouTubeVIS format and place it under the "imagenet/annotations/" directory.

Alternatively, you can refer to the instructions on generating pseudo-masks using MaskCut at [here](../../README.md#generating-annotations-for-imagenet-1k-with-maskcut). You'll need to convert these annotations into the [YouTubeVIS](https://competitions.codalab.org/competitions/20128) format (MaskCut provides MSCOCO format annotations). This format conversion is a necessary step to ensure compatibility with the training process of VideoCutLER.


## Expected dataset structure for [YouTubeVIS 2019](https://competitions.codalab.org/competitions/20128):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [YouTubeVIS 2021](https://competitions.codalab.org/competitions/28988):

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```
