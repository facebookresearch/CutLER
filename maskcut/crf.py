# Copyright (c) Meta Platforms, Inc. and affiliates.
# modfied by Xudong Wang based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py and third_party/TokenCut

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

MAX_ITER = 10
POS_W = 7 
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 50 
Bi_RGB_STD = 5

def densecrf(image, mask):
    h, w = mask.shape
    mask = mask.reshape(1, h, w)
    fg = mask.astype(np.float) 
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg,fg), axis=0))

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)
    
    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h,w)).astype(np.float32)
    return MAP
