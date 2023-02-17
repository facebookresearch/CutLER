# Copyright (c) Meta Platforms, Inc. and affiliates.
# merge all ImageNet annotation files as a single one.

import os
import json
import argparse

if __name__ == "__main__":
    # load model arguments
    parser = argparse.ArgumentParser(description='Merge json files')
    parser.add_argument('--base-dir', type=str,
                        default='annotations/',
                        help='Dir to the generated annotation files with MaskCut')
    parser.add_argument('--save-path', type=str, default="imagenet_train_fixsize480_tau0.15_N3.json",
                        help='Path to save the merged annotation file')
    # following arguments should be consistent with maskcut.py or maskcut_with_submitit.py (if use submitit)
    parser.add_argument('--num-folder-per-job', type=int, default=1,
                        help='Number of folders per json file')
    parser.add_argument('--fixed-size', type=int, default=480,
                        help='rescale the input images to a fixed size')
    parser.add_argument('--tau', type=float, default=0.15, help='threshold used for producing binary graph')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')

    args = parser.parse_args()

    base_name = 'imagenet_train_fixsize{}_tau{}_N{}'.format(args.fixed_size, args.tau, args.N)

    start_idx = 0
    every_k = args.num_folder_per_job
    missed_folders = []
    tobe_merged_ann_dicts = []

    # check if pseudo-masks for all 1000 ImageNet-1K folders are avaliable.
    while start_idx < 1000:
        end_idx = start_idx + every_k
        filename = "{}_{}_{}.json".format(base_name, start_idx, end_idx)
        tobe_merged = os.path.join(args.base_dir, filename)
        if not os.path.isfile(tobe_merged):
            end_idx = start_idx + 1
            tobe_merged_ = "{}_{}_{}.json".format(base_name, start_idx, end_idx)
            if not os.path.isfile(tobe_merged_):
                missed_folders.append(start_idx)
                start_idx += 1
                continue
            else:
                tobe_merged = tobe_merged_
                start_idx += 1
        else:
            start_idx += every_k
        tobe_merged_ann_dict = json.load(open(tobe_merged))
        tobe_merged_ann_dicts.append(tobe_merged_ann_dict)

    print("Warning: these folders are not found: ", missed_folders)

    # filter out repeated image info
    for idx, ann_dict in enumerate(tobe_merged_ann_dicts):
        images = []
        images_ids = []
        for image in ann_dict['images']:
            if image['id'] in images_ids:
                continue
            else:
                images.append(image)
                images_ids.append(image['id'])
        ann_dict['images'] = images

    # re-generate image_id and segment_id, and combine annotation info and image info
    # from all annotation files
    base_ann_dict = tobe_merged_ann_dicts[0]
    image_id = base_ann_dict['images'][-1]['id'] + 1
    segment_id = base_ann_dict['annotations'][-1]['id'] + 1
    segment_id_list = [ann['id'] for ann in base_ann_dict['annotations']]
    for tobe_merged_ann_dict in tobe_merged_ann_dicts[1:]:
        file_name_and_id = {}
        for i, image in enumerate(tobe_merged_ann_dict['images']):
            file_name_and_id[str(image['id'])] = image_id
            image['id'] = image_id
            base_ann_dict['images'].append(image)
            image_id = image_id + 1

        for i, annotation_info in enumerate(tobe_merged_ann_dict['annotations']):
            annotation_info["image_id"] = file_name_and_id[str(annotation_info["image_id"])]
            annotation_info["id"] = segment_id
            annotation_info["iscrowd"] = 0
            segment_id_list.append(segment_id)
            base_ann_dict['annotations'].append(annotation_info)
            segment_id = segment_id + 1

    segment_id = 1
    for ann in base_ann_dict['annotations']:
        ann["id"] = segment_id
        segment_id += 1

    # save the final json file.
    anns = [ann['id'] for ann in base_ann_dict['annotations']]
    anns_image_id = [ann['image_id'] for ann in base_ann_dict['annotations']]
    json.dump(base_ann_dict, open(args.save_path, 'w'))
    print("Done: {} images; {} anns.".format(len(base_ann_dict['images']), len(base_ann_dict['annotations'])))
