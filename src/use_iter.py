#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx

data_iter = mx.io.ImageRecordIter(
    path_imgrec='../input/amazon-from-space.rec',
    data_shape=(3, 256, 256),
    batch_size=16,
    rand_crop=True,
    rand_resize=True,
    rand_mirror=True,
    preprocess_threads=4,
    # Backend Parameter
    # Optional
    # Prefetch buffer size
    prefetch_buffer=4,
    # Backend Parameter,
    # Optional
    # Whether round batch,
    round_batch=True)

data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1, 2, 0)))
plt.show()
