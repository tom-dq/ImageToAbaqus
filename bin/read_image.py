from PIL import Image
import typing
import collections
import numpy
import scipy
import itertools
import random
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth


from matplotlib import pyplot as plt

SHOW_IMAGES = False
FN = r"data/spec3.png"

# OpenCV / scikit-learn

def read_raw_img(fn):

    img = cv2.imread(fn)

    # filter to reduce noise
    img = cv2.medianBlur(img, 3)

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = numpy.float32(flat_image)

    # meanshift
    # bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)

    # Rotate so it lines up???
    image_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # image_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_rot

def mean_shift(bandwidth, max_iter, img):
    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = numpy.float32(flat_image)

    # meanshift
    # bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=max_iter, bin_seeding=True, n_jobs=6)
    ms.fit(flat_image)
    labeled=ms.labels_

    # get number of segments
    segments = numpy.unique(labeled)
    print('Number of segments: ', segments.shape[0])
    n_segs = segments.shape[0]


    # get the average color of each segment
    total = numpy.zeros((segments.shape[0], 3), dtype=float)
    count = numpy.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = numpy.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))

    # Raw image of labels
    shape_2d = img.shape[0:2]
    raw_labels = labeled.reshape(shape_2d)

    if SHOW_IMAGES:
        cv2.imshow(f'{bandwidth} - {max_iter}: {n_segs}',result)

    return result, raw_labels


if __name__ == "__main__":
    img = read_raw_img(FN)

    # show the result
    if SHOW_IMAGES:
        cv2.imshow('orig', img)

    _, raw_labels = mean_shift(50, 2000, img)

    if SHOW_IMAGES:
        plt.imshow(raw_labels, interpolation='nearest')
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ImageRegion(typing.NamedTuple):
    colour: typing.Any
    pixels: int


def image_to_array(im: Image) -> numpy.array:
    """Go from pixels to distinct regions. One colour -> one value."""

    # Make greyscale first
    im_grey = im.convert("L")

    col_counts = sorted(collections.Counter(im_grey.getdata()).items())

    #num_getter = itertools.count(1000)
    #col_to_num = {col: next(num_getter) for col, _ in col_counts}

    # Make a numpy array with the same dimensions
    arr = numpy.zeros(shape=im_grey.size)

    # Slow!!! who cares.
    for ix in range(im_grey.size[0]):
        for iy in range(im_grey.size[1]):
            col = im_grey.getpixel((ix, iy))
            # num = col_to_num[col]
            arr[ix, iy] = col

    return arr





def clip_image(im: Image, remove_at_or_below: int):
    """Remove the smallest feature from an image"""

    col_counts = collections.Counter(im.getdata())

    small_cols = ...


def get_features(im: Image):
    """Find all the regions in a image which have the same colour
    and are touching."""

    # Get all the colours in an image
    col_counts = collections.Counter(im.getdata())
    for col, count in col_counts.most_common():
        yield col, count


if __name__ == "__main__":

    im = Image.open(FN)

    arr = image_to_array(im)

    plt.imshow(arr, interpolation='nearest')
    plt.show()
    _ = input()
