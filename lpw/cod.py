#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Gahan Saraiya
GiT: https://github.com/gahan9
StackOverflow: https://stackoverflow.com/users/story/7664524

Convert Imange in to matrix.
manipulate image
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
IMAGE_INPUT_DIR = os.path.join(BASE_DIR, "in")
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "out")
# initialize directory if not exist
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


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

    if(idx *3 < check*3){
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
    in_file = os.path.join(IMAGE_INPUT_DIR, IMAGES[0])
    out_file = os.path.join(IMAGE_OUTPUT_DIR, IMAGES[0])
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
   
    filter_function(gpu_pixel_array, total_pixels, block=block, grid=grid)
   
    result_array = numpy.empty_like(pixel_array)
    driver.memcpy_dtoh(result_array, gpu_pixel_array)
    # On monochrome images, Pixels are uint8 [0,255].
    # numpy.clip(bwPx, 0, 255, out=bwPx)
    # bwPx = bwPx.astype('uint8')
    result_array = (numpy.uint8(result_array))
    # pil_im = PILImage.fromarray(bwPx,mode ="RGB")
    status = save_image(out_file, result_array)
    return status


if __name__ == "__main__":
    IMAGES = os.listdir(IMAGE_INPUT_DIR)
    for image in IMAGES:
        manipulate_image(IMAGES, make_black_and_white)
