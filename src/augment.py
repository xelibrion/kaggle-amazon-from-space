from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt

file_name = '../input/train-jpg/train_32.jpg'
img = cv2.imread(file_name, -1)

seq = [
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
result = []

for i, aug in enumerate(seq):
    aug_r = aug.augment_image(img)
    result.append(aug_r)
    plt.subplot(1, 6, i + 1)
    plt.imshow(aug_r)

plt.subplot(1, 6, 6)
plt.imshow(img)
