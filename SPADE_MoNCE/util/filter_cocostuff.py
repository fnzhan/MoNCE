import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

paths = glob.glob("/mnt/lustre/fnzhan/datasets/CVPR2022/COCO-Stuff/train2017/*.jpg") # Your path for your dataset
sv_dir = '/mnt/lustre/fnzhan/datasets/CVPR2022/COCO-Stuff/subset/'
np.random.seed(123)
# paths_subset = np.random.choice(paths, 10000, replace=False) # choosing 1000 images randomly
# paths_subset = paths[:10000]
# rand_idxs = np.random.permutation(10_000)
train_paths = paths[:8000] # choosing the first 8000 as training set
val_paths = paths[8000:10000] # choosing last 2000 as validation set
# train_paths = paths_subset[train_idxs]
# val_paths = paths_subset[val_idxs]
# print(len(train_paths), len(val_paths))
n = 0
for path in train_paths:
    im = Image.open(path)
    im.save(path.replace('train2017', 'subset/train'))
    n += 1
    print(n)

for path in val_paths:
    im = Image.open(path)
    im.save(path.replace('train2017', 'subset/test'))
    n += 1
    print(n)