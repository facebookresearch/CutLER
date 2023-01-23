# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py and https://github.com/NVlabs/FreeSOLO/tree/main/freesolo/engine/trainer.py

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import numpy as np
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import copy
import random
import torch.nn.functional as F
from detectron2.structures.instances import Instances
from detectron2.structures import BitMasks

from detectron2.engine import SimpleTrainer

__all__ = ["CustomSimpleTrainer", "CustomAMPTrainer"]

class CustomSimpleTrainer(SimpleTrainer):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, cfg=None, use_copy_paste=False, 
                copy_paste_rate=-1, copy_paste_random_num=None, copy_paste_min_ratio=-1, 
                copy_paste_max_ratio=-1, visualize_copy_paste=False):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__(model, data_loader, optimizer)

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        self.cfg = cfg
        # model.train()

        # self.model = model
        # self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        # self._data_loader_iter_obj = None
        # self.optimizer = optimizer

        self.use_copy_paste = use_copy_paste if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE
        self.cfg_COPY_PASTE_RATE = copy_paste_rate if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_RATE
        self.cfg_COPY_PASTE_RANDOM_NUM = copy_paste_random_num if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_RANDOM_NUM
        self.cfg_COPY_PASTE_MIN_RATIO = copy_paste_min_ratio if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_MIN_RATIO
        self.cfg_COPY_PASTE_MAX_RATIO = copy_paste_max_ratio if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_MAX_RATIO
        self.cfg_VISUALIZE_COPY_PASTE = visualize_copy_paste if self.cfg is None else self.cfg.DATALOADER.VISUALIZE_COPY_PASTE

    def IoU(self, mask1, mask2): # only work when the batch size is 1
        mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
        intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
        union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
        return (intersection.to(torch.float) / union).mean().view(1, -1)

    def IoY(self, mask1, mask2): # only work when the batch size is 1
        # print(mask1.size(), mask2.size())
        mask1, mask2 = mask1.squeeze(), mask2.squeeze()
        mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
        intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
        union = torch.sum(mask2, dim=[-1, -2]).squeeze()
        return (intersection.to(torch.float) / union).mean().view(1, -1)

    def copy_and_paste(self, labeled_data, unlabeled_data):
        new_unlabeled_data = []
        def mask_iou_matrix(x, y, mode='iou'):
            x = x.reshape(x.shape[0], -1).float() 
            y = y.reshape(y.shape[0], -1).float()
            inter_matrix = x @ y.transpose(1, 0) # n1xn2
            sum_x = x.sum(1)[:, None].expand(x.shape[0], y.shape[0])
            sum_y = y.sum(1)[None, :].expand(x.shape[0], y.shape[0])
            if mode == 'ioy':
                iou_matrix = inter_matrix / (sum_y) # [1, 1]
            else:
                iou_matrix = inter_matrix / (sum_x + sum_y - inter_matrix) # [1, 1]
            return iou_matrix

        def visualize_data(data, save_path = './sample.jpg'):
            from data import detection_utils as utils 
            from detectron2.data import DatasetCatalog, MetadataCatalog
            from detectron2.utils.visualizer import Visualizer 
            data["instances"] = data["instances"].to(device='cpu')
            img = data["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, 'RGB')
            metadata = MetadataCatalog.get('imagenet_train_tau0.15')
            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = data["instances"].get_fields() 
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes"), # ("gt_boxes", None),
                    masks=target_fields.get("gt_masks"), # ("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
            )
            print("Saving to {} ...".format(save_path))
            vis.save(save_path)

        for cur_labeled_data, cur_unlabeled_data in zip(labeled_data, unlabeled_data):
            cur_labeled_instances = cur_labeled_data["instances"]
            cur_labeled_image = cur_labeled_data["image"]
            cur_unlabeled_instances = cur_unlabeled_data["instances"]
            cur_unlabeled_image = cur_unlabeled_data["image"]

            num_labeled_instances = len(cur_labeled_instances)
            copy_paste_rate = random.random()
            
            if self.cfg_COPY_PASTE_RATE >= copy_paste_rate and num_labeled_instances > 0:
                if self.cfg_COPY_PASTE_RANDOM_NUM:
                    num_copy = 1 if num_labeled_instances == 1 else np.random.randint(1, max(1, num_labeled_instances))
                else:
                    num_copy = num_labeled_instances
            else:
                num_copy = 0
            if num_labeled_instances == 0 or num_copy == 0:
                new_unlabeled_data.append(cur_unlabeled_data)
            else:
                # print("num_labeled_instances, num_copy: ", num_labeled_instances, num_copy)
                choice = np.random.choice(num_labeled_instances, num_copy, replace=False)
                copied_instances = cur_labeled_instances[choice].to(device=cur_unlabeled_instances.gt_boxes.device)
                copied_masks = copied_instances.gt_masks
                copied_boxes = copied_instances.gt_boxes
                _, labeled_h, labeled_w = cur_labeled_image.shape
                _, unlabeled_h, unlabeled_w = cur_unlabeled_image.shape

                # rescale the labeled image to align with unlabeled one.
                if isinstance(copied_masks, torch.Tensor):
                    masks_new = copied_masks[None, ...].float()
                else:
                    masks_new = copied_masks.tensor[None, ...].float()
                # resize the masks with a random ratio from 0.5 to 1.0
                resize_ratio = random.uniform(self.cfg_COPY_PASTE_MIN_RATIO, self.cfg_COPY_PASTE_MAX_RATIO)
                w_new = int(resize_ratio * unlabeled_w)
                h_new = int(resize_ratio * unlabeled_h)

                w_shift = random.randint(0, unlabeled_w - w_new)
                h_shift = random.randint(0, unlabeled_h - h_new)

                cur_labeled_image_new = F.interpolate(cur_labeled_image[None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).byte().squeeze(0)
                if isinstance(copied_masks, torch.Tensor):
                    masks_new = F.interpolate(copied_masks[None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).bool().squeeze(0)
                else:
                    masks_new = F.interpolate(copied_masks.tensor[None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).bool().squeeze(0)
                copied_boxes.scale(1. * unlabeled_w / labeled_w * resize_ratio, 1. * unlabeled_h / labeled_h * resize_ratio)

                if isinstance(cur_unlabeled_instances.gt_masks, torch.Tensor):
                    _, mask_w, mask_h = cur_unlabeled_instances.gt_masks.size()
                else:
                    _, mask_w, mask_h = cur_unlabeled_instances.gt_masks.tensor.size()
                masks_new_all = torch.zeros(num_copy, mask_w, mask_h)
                image_new_all = torch.zeros_like(cur_unlabeled_image)

                image_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += cur_labeled_image_new
                masks_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += masks_new

                cur_labeled_image = image_new_all.byte() #.squeeze(0)
                if isinstance(copied_masks, torch.Tensor):
                    copied_masks = masks_new_all.bool() #.squeeze(0)
                else:
                    copied_masks.tensor = masks_new_all.bool() #.squeeze(0)
                copied_boxes.tensor[:, 0] += h_shift
                copied_boxes.tensor[:, 2] += h_shift
                copied_boxes.tensor[:, 1] += w_shift
                copied_boxes.tensor[:, 3] += w_shift

                copied_instances.gt_masks = copied_masks
                copied_instances.gt_boxes = copied_boxes
                copied_instances._image_size = (unlabeled_h, unlabeled_w)
                if len(cur_unlabeled_instances) == 0:
                    if isinstance(copied_instances.gt_masks, torch.Tensor):
                        alpha = copied_instances.gt_masks.sum(0) > 0
                    else:
                        alpha = copied_instances.gt_masks.tensor.sum(0) > 0
                    # merge image
                    alpha = alpha.cpu()
                    composited_image = (alpha * cur_labeled_image) + (~alpha * cur_unlabeled_image)
                    cur_unlabeled_data["image"] = composited_image
                    cur_unlabeled_data["instances"] = copied_instances
                else:
                    # remove the copied object if iou greater than 0.5
                    if isinstance(copied_masks, torch.Tensor):
                        iou_matrix = mask_iou_matrix(copied_masks, cur_unlabeled_instances.gt_masks, mode='ioy') # nxN
                    else:
                        iou_matrix = mask_iou_matrix(copied_masks.tensor, cur_unlabeled_instances.gt_masks.tensor, mode='ioy') # nxN

                    keep = iou_matrix.max(1)[0] < 0.5
                    if keep.sum() == 0:
                        new_unlabeled_data.append(cur_unlabeled_data)
                        continue
                    copied_instances = copied_instances[keep]
                    # update existing instances in unlabeled image
                    if isinstance(copied_instances.gt_masks, torch.Tensor):
                        alpha = copied_instances.gt_masks.sum(0) > 0
                        cur_unlabeled_instances.gt_masks = ~alpha * cur_unlabeled_instances.gt_masks
                        areas_unlabeled = cur_unlabeled_instances.gt_masks.sum((1,2))
                    else:
                        alpha = copied_instances.gt_masks.tensor.sum(0) > 0
                        cur_unlabeled_instances.gt_masks.tensor = ~alpha * cur_unlabeled_instances.gt_masks.tensor
                        areas_unlabeled = cur_unlabeled_instances.gt_masks.tensor.sum((1,2))
                    # merge image
                    alpha = alpha.cpu()
                    composited_image = (alpha * cur_labeled_image) + (~alpha * cur_unlabeled_image)
                    # merge instances
                    merged_instances = Instances.cat([cur_unlabeled_instances[areas_unlabeled > 0], copied_instances])
                    # update boxes
                    if isinstance(merged_instances.gt_masks, torch.Tensor):
                        merged_instances.gt_boxes = BitMasks(merged_instances.gt_masks).get_bounding_boxes()
                        # merged_instances.gt_boxes = merged_instances.gt_masks.get_bounding_boxes()
                    else:
                        merged_instances.gt_boxes = merged_instances.gt_masks.get_bounding_boxes()

                    cur_unlabeled_data["image"] = composited_image
                    cur_unlabeled_data["instances"] = merged_instances
                if self.cfg_VISUALIZE_COPY_PASTE:
                    visualize_data(cur_unlabeled_data, save_path = 'sample_{}.jpg'.format(np.random.randint(5)))
                new_unlabeled_data.append(cur_unlabeled_data)
        return new_unlabeled_data

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        # print(data, len(data))
        if self.use_copy_paste:
            # print('using copy paste')
            data = self.copy_and_paste(copy.deepcopy(data[::-1]), data)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if not torch.isnan(losses):
            self.optimizer.zero_grad()
            losses.backward()
        else:
            print('Nan loss. Skipped.')

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


class CustomAMPTrainer(CustomSimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, cfg=None, grad_scaler=None, use_copy_paste=False, 
                copy_paste_rate=-1, copy_paste_random_num=None, copy_paste_min_ratio=-1, 
                copy_paste_max_ratio=-1, visualize_copy_paste=False):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, cfg=cfg, use_copy_paste=use_copy_paste, \
            copy_paste_rate=copy_paste_rate, copy_paste_random_num=copy_paste_random_num, \
            copy_paste_min_ratio=copy_paste_min_ratio, copy_paste_max_ratio=copy_paste_max_ratio, \
            visualize_copy_paste=visualize_copy_paste)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        if self.use_copy_paste:
            # print('using copy paste')
            data = self.copy_and_paste(copy.deepcopy(data[::-1]), data)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not torch.isnan(losses):
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()
        else:
            print('Nan loss.')

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
