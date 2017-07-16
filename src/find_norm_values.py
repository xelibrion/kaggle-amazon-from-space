#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
all_mean = []

for root, dir, files in os.walk('../input/train-jpg'):
    for f in tqdm(files):
        img = cv2.imread(os.path.join(root, f), 1)
        all_mean.append(np.mean(img, axis=(0, 1)))

print(pd.DataFrame(all_mean).agg(['mean', 'std']) / 255)
