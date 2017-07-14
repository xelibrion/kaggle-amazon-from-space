from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt


def expand_versions(image):
    crop_seq = [
        # leave top-left
        iaa.Crop(px=(0, 32, 32, 0), keep_size=False),
        # top-right
        iaa.Crop(px=(0, 0, 32, 32), keep_size=False),
        # bottom-left
        iaa.Crop(px=(32, 32, 0, 0), keep_size=False),
        # bottom-right
        iaa.Crop(px=(32, 0, 0, 32), keep_size=False),
        # leave center
        iaa.Crop(px=(16, 16, 16, 16), keep_size=False),
    ]
    mirror_transform = iaa.Fliplr(1.0)

    crops = [x.augment_image(image) for x in crop_seq]
    mirrors = [mirror_transform.augment_image(x) for x in crops]

    crops.extend(mirrors)

    return crops


def main():

    file_name = '../input/train-jpg/train_32.jpg'
    img = cv2.imread(file_name, -1)

    for i, aug in enumerate(expand_versions(img)):
        plt.subplot(1, 10, i + 1)
        plt.imshow(aug)
