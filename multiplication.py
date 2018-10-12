#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Gahan Saraiya
GiT: https://github.com/gahan9
StackOverflow: https://stackoverflow.com/users/story/7664524
"""
import pycuda.autoinit
from pycuda import driver
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

__global__ void addition(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}
""")


class Multiply(object):
    def __init__(self):
        self.a = numpy.random.randn(400).astype(numpy.float32)
        self.b = numpy.random.randn(400).astype(numpy.float32)
        self.dest = numpy.zeros_like(self.a)

    def multiply(self):
        multiply = mod.get_function("multiply")
        multiply(
            driver.Out(self.dest), driver.In(self.a), driver.In(self.b),
            block=(400, 1, 1), grid=(1, 1))

    def addition(self):
        addition = mod.get_function("addition")
        addition(
            driver.Out(self.dest), driver.In(self.a), driver.In(self.b),
            block=(400, 1, 1), grid=(1, 1))


def test():
    m = Multiply()
    m.multiply()
    print("Result:----\n", m.dest)
    print("Verifying....")
    print(m.dest - m.a*m.b)


if __name__ == "__main__":
    test()
