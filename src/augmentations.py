from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import random


class D4(object):
    """Random transformation from D4 group
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = np.transpose(img, [1, 0, 2])

        if random.random() < 0.5:
            img = np.flipud(img)

        if random.random() < 0.5:
            img = np.fliplr(img)

        return np.copy(img)


class Add(object):
    """Adds values within a range

    per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.
    """

    def __init__(self, from_add=-10, to_add=+10, per_channel=0.5):
        self.from_add = from_add
        self.to_add = to_add
        self.per_channel = per_channel

    def __call__(self, img):
        img_array = np.array(img)

        adder = iaa.Add(
            (self.from_add, self.to_add),
            per_channel=self.per_channel, )

        img_n = adder.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class ContrastNormalization(object):
    """Changes contrast
    """

    def __init__(self, contrast_from=0.9, contrast_to=1.1, per_channel=0.5):
        self.contrast_from = contrast_from
        self.contrast_to = contrast_to
        self.per_channel = per_channel

    def __call__(self, img):
        img_array = np.array(img)

        contrastor = iaa.ContrastNormalization(
            (self.contrast_from, self.contrast_to), self.per_channel)
        img_n = contrastor.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')
