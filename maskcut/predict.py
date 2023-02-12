"""
download pretrained weights to ./weights
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth
"""

import sys

sys.path.append("maskcut")
import numpy as np
import PIL.Image as Image
import torch
from scipy import ndimage
from colormap import random_color

import dino
from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut import maskcut

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # DINO pre-trained model
        vit_features = "k"
        self.patch_size = 8
        # adapted dino.ViTFeat to load from local pretrained_path
        self.backbone_base = dino.ViTFeat(
            "weights/dino_vitbase8_pretrain.pth",
            768,
            "base",
            vit_features,
            self.patch_size,
        )

        self.backbone_small = dino.ViTFeat(
            "weights/dino_deitsmall8_300ep_pretrain.pth",
            384,
            "small",
            vit_features,
            self.patch_size,
        )
        self.backbone_base.eval()
        self.backbone_base.cuda()
        self.backbone_small.eval()
        self.backbone_small.cuda()

    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        model: str = Input(
            description="Choose the model architecture",
            default="base",
            choices=["small", "base"]
        ),
        n_pseudo_masks: int = Input(
            description="The maximum number of pseudo-masks per image",
            default=3,
        ),
        tau: float = Input(
            description="Threshold used for producing binary graph",
            default=0.15,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        backbone = self.backbone_base if model == "base" else self.backbone_small

        # MaskCut hyperparameters
        fixed_size = 480

        # get pesudo-masks with MaskCut
        bipartitions, _, I_new = maskcut(
            str(image),
            backbone,
            self.patch_size,
            tau,
            N=n_pseudo_masks,
            fixed_size=fixed_size,
            cpu=False,
        )

        I = Image.open(str(image)).convert("RGB")
        width, height = I.size
        pseudo_mask_list = []
        for idx, bipartition in enumerate(bipartitions):
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(pseudo_mask).cuda()

            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            pseudo_mask_list.append(pseudo_mask)

        out = np.array(I)
        for pseudo_mask in pseudo_mask_list:

            out = vis_mask(out, pseudo_mask, random_color(rgb=True))

        output_path = f"/tmp/out.png"

        out.save(str(output_path))

        return Path(output_path)


def vis_mask(input, mask, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)
