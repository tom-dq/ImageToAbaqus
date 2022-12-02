from PIL import Image
import typing
import collections
import numpy
import scipy
import itertools
import random

from matplotlib import pyplot as plt


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
    fn = r"C:\Users\Tom Wilson\source\ImageToAbaqus\data\spec3.png"

    im = Image.open(fn)

    arr = image_to_array(im)

    plt.imshow(arr, interpolation='nearest')
    plt.show()
