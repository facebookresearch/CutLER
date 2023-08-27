# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from detectron2 and cocoapi

import argparse
import os

from mask2former_video.data_video.datasets.ytvis_api.ytvoseval import YTVOSeval
from mask2former_video.data_video.datasets.ytvis_api.ytvos import YTVOS

def print_and_summary(cocoEval):
    str_print = ""
    for key in cocoEval.stats:
        str_print += "{:.2f},".format(key*100)
    return str_print

def get_parser():
    parser = argparse.ArgumentParser(description="eval configs")
    parser.add_argument(
        "--dataset-path", default="DATASETS", help="path to the annotation file",
    )
    parser.add_argument(
        "--dataset-name", default="ytvis_2019", help="path to the annotation file", 
    )
    parser.add_argument(
        "--result-path", default="OUTPUT", help="path to the the result file", 
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    annFile = os.path.join(args.dataset_path, args.dataset_name, 'train.json')
    cocoGt=YTVOS(annFile)

    resFile = os.path.join(args.result_path, 'inference/results.json')
    cocoDt=cocoGt.loadRes(resFile)

    annType = 'segm'
    print('Running demo for {} results.'.format(annType))
    cocoEval = YTVOSeval(cocoGt,cocoDt,annType)
    cocoEval.params.useCats  = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    copypaste = print_and_summary(cocoEval)
    print(copypaste)