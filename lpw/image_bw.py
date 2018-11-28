#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Gahan Saraiya
GiT: https://github.com/gahan9
StackOverflow: https://stackoverflow.com/users/story/7664524

Convert Imange in to matrix.
manipulate image
i.e. Making image black and white
"""

import os
import numpy

from pycuda import driver, compiler, gpuarray, tools
# libraries to read image
import matplotlib.image as img
import PIL
from PIL import Image as PILImage
# -- initialize the device
import pycuda.autoinit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "in")


kernel = """
__global__ void black_and_white(float *image, int check){
    /*To scale RGB colors a matrix like this is used:
        float mat[4][4] = {
            rscale, 0.0,    0.0,    0.0,
            0.0,    gscale, 0.0,    0.0,
            0.0,    0.0,    bscale, 0.0,
            0.0,    0.0,    0.0,    1.0,
        };
        Where rscale, gscale, and bscale specify how much to scale the r, g, and b components of colors. This can be used to alter the color balance of an image.
        In effect, this calculates:
            tr = r*rscale;
            tg = g*gscale;
            tb = b*bscale;
    */
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    // work only if pixels are in range .. the check number here is total number of pixels
    if(idx*3 < check*3){
        int val = 0.21 * image[idx*3] + 0.71 * image[idx*3+1] + 0.07 * image[idx*3+2];
        image[idx*3] = val;
        image[idx*3+1] = val;
        image[idx*3+2] = val;
    }
}
"""
# compile the kernel code
mod = compiler.SourceModule(kernel)
# get the kernel function from the compiled module
make_black_and_white = mod.get_function("black_and_white")

def open_image(image_path, flag=0):
    if flag: # use matplotlib
        image = img.imread(image_path)
    else: # use PIL
        image = PILImage.open(image_path)
    return image

def save_image(image_path, array):
    status = img.imsave(image_path, array)
    return True

def manipulate_image(image_name, filter_function):
    in_file = os.path.join(IMAGE_DIR, image_name)
    image_ext = image_name.split('.')[-1]
    image_out_name = ''.join(image_name.split('.')[:-1]) + '_bw'
    out_file = os.path.join(IMAGE_DIR, image_out_name)
    image = open_image(in_file) # open image
    # store image pixels to numpy array of data type float32
    pixel_array = numpy.array(image).astype(numpy.float32)    
    total_pixels = numpy.int32(image.size[0]*image.size[1])
    print(
        "Size: {}"
        "\nTotal Pixels: {}".format(image.size, total_pixels)
    )

    # Reserve memory in GPU to store pixel_array
    gpu_pixel_array = driver.mem_alloc(pixel_array.nbytes)
    # Copy pixel_array from the Python buffer to the device pointer(GPU) gpu_pixel_array
    driver.memcpy_htod(gpu_pixel_array, pixel_array)
    
    # define block size
    BLOCK_SIZE = 1024
    block = (BLOCK_SIZE, 1, 1)
    # construct grid to efficiently allocate workers according to block size
    grids = int(image.size[0] * image.size[1]/BLOCK_SIZE) + 1
    grid = (grids,1,1)
    print(
        "Block: {}"
        "\ngrids: {}"
        "\nBLOCK_SIZE x grids: {}"
        "\nGrid: {}".format(block, grids, grids*BLOCK_SIZE, grid)
    )

    # apply filter function on matrix
    filter_function(gpu_pixel_array, total_pixels, block=block, grid=grid)

    # construct an empty array to store computed result by gpu
    result_array = numpy.empty_like(pixel_array)
    # copy filtered matrix to host memory buffer from device (GPU)
    driver.memcpy_dtoh(result_array, gpu_pixel_array)

    # convert result array in to uint8 data type (unsigned 8 bit integer).
    # as it is the range of pixel.
    result_array = numpy.uint8(result_array)
    
    # save array as image to disk
    status = save_image(out_file, result_array)
    print("-"*50)
    print("Image {} converted and saved at {}".format(image_name, out_file))
    print("-"*50)
    return status


if __name__ == "__main__":
    IMAGES = os.listdir(IMAGE_DIR)
    for i in IMAGES:
        manipulate_image(i, make_black_and_white)
