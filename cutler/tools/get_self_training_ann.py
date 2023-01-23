#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import tqdm
import torch
import datetime
import argparse
import pycocotools.mask as cocomask
from detectron2.utils.file_io import PathManager

INFO = {
    "description": "ImageNet-1K: Self-train",
    "url": "",
    "version": "1.0",
    "year": 2022,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

new_dict_filtered = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

category_info = {
    "is_crowd": 0,
    "id": 1
}


def segmToRLE(segm, h, w):
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask

def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask

def BatchIoU(masks1, masks2):
    n1, n2 = masks1.size()[0], masks2.size()[0]
    masks1, masks2 = (masks1>0.5).to(torch.bool), (masks2>0.5).to(torch.bool)
    masks1_ = masks1[:,None,:,:,].expand(-1, n2, -1, -1)
    masks2_ = masks2[None,:,:,:,].expand(n1, -1, -1, -1)

    intersection = torch.sum(masks1_ * (masks1_ == masks2_), dim=[-1, -2])
    union = torch.sum(masks1_ + masks2_, dim=[-1, -2])
    ious = intersection.to(torch.float) / union
    return ious

if __name__ == "__main__":
    # load model arguments
    parser = argparse.ArgumentParser(description='Generate json files for the self-training')
    parser.add_argument('--new-pred', type=str, 
                        default='output/inference/coco_instances_results.json',
                        help='Path to model predictions')
    parser.add_argument('--prev-ann', type=str, 
                        default='DETECTRON2_DATASETS/imagenet/annotations/cutler_imagenet1k_train.json',
                        help='Path to annotations in the previous round')
    parser.add_argument('--save-path', type=str,
                        default='DETECTRON2_DATASETS/imagenet/annotations/cutler_imagenet1k_train_r1.json',
                        help='Path to save the generated annotation file')
    # parser.add_argument('--n-rounds', type=int, default=1,
    #                     help='N-th round of self-training')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence score thresholds')
    args = parser.parse_args()

    # load model predictions
    new_pred = args.new_pred
    with PathManager.open(new_pred, "r") as f:
        predictions = json.load(f)

    # filter out low-confidence model predictions
    THRESHOLD = args.threshold
    pred_image_to_anns = {}
    for id, ann in enumerate(predictions):
        confidence_score = ann['score']
        if confidence_score >= THRESHOLD:
            if ann['image_id'] in pred_image_to_anns:
                pred_image_to_anns[ann['image_id']].append(ann)
            else:
                pred_image_to_anns[ann['image_id']] = [ann]

    # load psedu-masks used by the previous round
    pseudo_ann_dict = json.load(open(args.prev_ann))
    pseudo_image_list = pseudo_ann_dict['images']
    pseudo_annotations = pseudo_ann_dict['annotations']

    pseudo_image_to_anns = {}
    for id, ann in enumerate(pseudo_annotations):
        if ann['image_id'] in pseudo_image_to_anns:
            pseudo_image_to_anns[ann['image_id']].append(ann)
        else:
            pseudo_image_to_anns[ann['image_id']] = [ann]

    # merge model predictions and the json file used by the previous round.
    merged_anns = []
    num_preds, num_pseudo = 0, 0
    for k, anns_pseudo in tqdm.tqdm(pseudo_image_to_anns.items()):
        masks = []
        for ann in anns_pseudo:
            segm = ann['segmentation']
            mask = cocosegm2mask(segm, segm['size'][0], segm['size'][1])
            masks.append(torch.from_numpy(mask))
        pseudo_masks = torch.stack(masks, dim=0).cuda()
        del masks
        num_pseudo += len(anns_pseudo)
        try:
            anns_pred = pred_image_to_anns[k]
        except:
            merged_anns += anns_pseudo
            continue
        masks = []
        for ann in anns_pred:
            segm = ann['segmentation']
            mask = cocosegm2mask(segm, segm['size'][0], segm['size'][1])
            masks.append(torch.from_numpy(mask))
        pred_masks = torch.stack(masks, dim=0).cuda()
        num_preds += len(anns_pred)
        try:
            ious = BatchIoU(pseudo_masks, pred_masks)
            iou_max, _ = ious.max(dim=1)
            selected_index = (iou_max < 0.5).nonzero()
            selected_pseudo = [anns_pseudo[i] for i in selected_index]
            merged_anns += anns_pred + selected_pseudo
            # if num_preds % 200000 == 0:
            #     print(len(merged_anns), num_preds, num_pseudo)
        except:
            merged_anns += anns_pseudo

    for key in pred_image_to_anns:
        if key in pseudo_image_to_anns:
            continue
        else:
            merged_anns += pred_image_to_anns[key]

    # re-generate annotation id
    ann_id = 1
    for ann in merged_anns:
        ann['id'] = ann_id
        ann['area'] = ann['bbox'][-1] * ann['bbox'][-2]
        ann['iscrowd'] = 0
        ann['width'] = ann['segmentation']['size'][0]
        ann['height'] = ann['segmentation']['size'][1]
        ann_id += 1

    new_dict_filtered['images'] = pseudo_image_list
    new_dict_filtered['annotations'] = merged_anns

    # save annotation file
    # save_path = os.path.join(args.save_path, "cutler_imagenet1k_train_r{}.json".format(args.n_rounds))
    json.dump(new_dict_filtered, open(args.save_path, 'w'))
    print("Done: {} images; {} anns.".format(len(new_dict_filtered['images']), len(new_dict_filtered['annotations'])))