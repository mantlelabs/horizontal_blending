#!/usr/bin/env python3

import argparse
import sys
import uuid
import os
import shutil
import atexit
import subprocess
import rasterio
import numpy as np
import cv2
import math


def parse_inputs():

    parser = argparse.ArgumentParser(
        description='horizontal seamless mosaicking of two images')
    parser.add_argument('-nd_A', help='NoData value for Image A',
                        type=int, default=0)
    parser.add_argument('-nd_B', help='NoData value for Image B',
                        type=int, default=0)
    parser.add_argument('-o', help='/path/to/output_file',
                        type=str, default=None)

    parser.add_argument('-b', help='size of border to remove (in pixel)',
                        type=int, default=0)

    parser.add_argument('image_A', help='Image A', type=str)
    parser.add_argument('image_B', help='Image B', type=str)

    #  exit with error in case of no arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    else:
        args = parser.parse_args()

        return args


def pad_with(vector, pad_width, iaxis, kwargs):
    """helper function to pad image with a value

    from https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html"""

    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

    return vector


def remove_padding(matrix: np.ndarray,
                   size: int) -> np.ndarray:
    """helper function to remove padding of size 'size'"""
    return matrix[size:matrix.shape[0]-size, size:matrix.shape[1]-size]


def merge_images(images_list: list,
                 tmp_folder: str,
                 nodatas: list) -> str:

    #  build file name of merged image out of inputs
    merged_file = os.path.join(tmp_folder, '{}_{}'.format(
        os.path.splitext(os.path.basename(images_list[0]))[0],
        os.path.basename(images_list[1])))

    #  call gdalmerge
    cmd = ['gdal_merge.py', '-separate',
           '-init', '{0} {1}'.format(nodatas[0], nodatas[1]),
           '-o', merged_file] + images_list
    subprocess.check_call(cmd)

    return merged_file


def remove_edges(images: list,
                 buffer_size_pixel: int,
                 nodatas: list,
                 tmp_folder: str) -> list:

    #  create kernel for dilation
    kernel = np.ones((buffer_size_pixel*2+1, buffer_size_pixel*2+1), np.uint8)

    #  loop through both images
    out_imgs = []
    for i in [0, 1]:

        #  load img data
        with rasterio.open(images[i], 'r') as img:
            kwds = img.profile
            img_data = img.read(1)

        #  generate binary nodata/data mask
        mask = img_data == nodatas[i]

        #  set same dtype as input image
        mask = mask.astype(kwds['dtype'])

        #  pad mask with ones
        mask = np.pad(mask, buffer_size_pixel, pad_with)

        #  do dilation
        mask_dil = cv2.dilate(mask, kernel, iterations=1)

        #  remove padding
        mask_dil = remove_padding(mask_dil, buffer_size_pixel)

        #  overwrite data pixels of dilated mask with original image
        #  and nodata pixels with nodata value
        mask_dil[mask_dil == 0] = img_data[mask_dil == 0]
        mask_dil[mask_dil == 1] = nodatas[i]

        #  generate filename
        filename, ext = os.path.splitext(os.path.basename(images[i]))
        img_out = os.path.join(tmp_folder,
                               '{}_buffered{}'.format(filename, ext))
        out_imgs.append(img_out)

        with rasterio.open(img_out, 'w', **kwds) as img:
            img.write(np.reshape(mask_dil, [1] + list(mask_dil.shape)))

    return out_imgs


def sigmoid(num: int) -> np.ndarray:
    """sigmoid function for creating alpha mask
    from https://stackoverflow.com/questions/29106702/blend-overlapping-images-in-python"""  # noqa E501

    x = np.arange(-10, 10, 2/int(num))[:-1:10]

    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1 / (1 + math.exp(-x[i]))
    return y


def get_overlap_area(image_data_A: np.ndarray,
                     image_data_B: np.ndarray,
                     nodata_A: int,
                     nodata_B: int) -> np.ndarray:

    valid_A = image_data_A != nodata_A
    valid_B = image_data_B != nodata_B

    return np.logical_and(valid_A, valid_B)


def blend_row(row_A: np.ndarray,
              row_B: np.ndarray,
              row_overlap: np.ndarray) -> np.ndarray:

    len_overlap = np.sum(row_overlap.astype(np.uint32))
    alpha = sigmoid(len_overlap)

    print(alpha)

    out = row_A[row_overlap == 1] * (1.0 - alpha) + \
        row_B[row_overlap == 1] * alpha

    out_row = row_overlap.astype(row_A.dtype)

    out_row[row_overlap == 1] = out

    return out_row


def do_blending(merged_image: str,
                nodata_A: int,
                nodata_B: int) -> np.ndarray:
    """
    loop horizontally over image and do row-wise blending
    """

    with rasterio.open(merged_image, 'r') as img:
        kwds = img.profile
        image_data_A = img.read(1)
        image_data_B = img.read(2)

    overlap = get_overlap_area(image_data_A, image_data_B,
                               nodata_A, nodata_B)

    #  prepare output image and populate
    merged_img = image_data_A
    merged_img[image_data_A == nodata_A] = \
        image_data_B[image_data_A == nodata_A]
    merged_img[overlap] = 0

    #  loop over rows of image and apply sigmoid blend to overlap
    for r in range(image_data_A.shape[0]):
        merged_img[r, :] = merged_img[r, :] + \
            blend_row(image_data_A[r, :],
                      image_data_B[r, :],
                      overlap[r, :])

    return merged_img, kwds


def horizontal_blending(image_A: str,
                        image_B: str,
                        output_file: str = None,
                        nodata_A: int = 0,
                        nodata_B: int = 0,
                        buffer_size_pixel: int = None) -> str:
    """merge two images and blend overlapping areas"""

    #  handle tmp folder
    tmp_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              uuid.uuid4().hex)
    os.makedirs(tmp_folder)
    atexit.register(shutil.rmtree, tmp_folder)

    #  1. buffer nodata if necessary to remove noisy edges
    if buffer_size_pixel > 0:
        images_list = remove_edges([os.path.realpath(image_A),
                                    os.path.realpath(image_B)],
                                   buffer_size_pixel,
                                   [nodata_A, nodata_B],
                                   tmp_folder)
    else:
        images_list = [os.path.realpath(image_A),
                       os.path.realpath(image_B)]

    #  2. merge images to get correct geometries
    merged_image = merge_images(images_list, tmp_folder,
                                [nodata_A, nodata_B])

    #  3. do the blending
    result, kwds = do_blending(merged_image, nodata_A, nodata_B)

    if output_file is None:
        output_file = 'merged.tif'

    kwds['count'] = 1
    with rasterio.open(output_file, 'w', **kwds) as img:
        img.write(np.reshape(result, [1] + list(result.shape)))


if __name__ == "__main__":
    args = parse_inputs()

    horizontal_blending(args.image_A,
                        args.image_B,
                        args.o,
                        args.nd_A,
                        args.nd_B,
                        args.b)
