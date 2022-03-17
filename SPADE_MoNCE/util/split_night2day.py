"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from PIL import Image
import numpy as np

bs_dir = '/mnt/lustre/fnzhan/datasets/CVPR2022/night2day/'
sv_dir = bs_dir + 'split/test/'

im_dir = bs_dir + 'test/'

nms = os.listdir(im_dir)
# n = 0
for nm in nms:
    im_path = im_dir + nm
    im = Image.open(im_path)
    im = np.array(im)
    inp, gt = im[:, 0:256, :], im[:, 256:, :]
    inp, gt = Image.fromarray(inp), Image.fromarray(gt)
    inp.save(sv_dir + 'input/' + nm)
    gt.save(sv_dir + 'gt/' + nm)


