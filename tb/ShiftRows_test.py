#!/usr/bin/env python

import time
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class ShiftRowsTest:
    def __init__(self):
        self.getSourceModule()

    def getSourceModule(self):
        file = open("../kernels/ShiftRows.cuh", "r")
        kernelwrapper = file.read()
        file.close()
        self.module = SourceModule(kernelwrapper)


    def shiftrows_gpu(self, message, length):
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()

        # Start recording execution time
        start.record()

        # Device memory allocation for input and output arrays
        io_message_gpu = cuda.mem_alloc_like(message)

        # Copy data from host to device
        cuda.memcpy_htod(io_message_gpu, message)

        # Call the kernel function from the compiled module
        prg = self.module.get_function("ShiftRowsTest")

        # Calculate block size and grid size
        block_size = length
        grid_size = 1
        if (block_size > 1024):
            block_size = 1024
            grid_size = (length - 1) / 1024 + 1;

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # Call the kernel loaded to the device
        prg(io_message_gpu, np.uint32(length), block=blockDim, grid=gridDim)

        # Copy result from device to the host
        res = np.empty_like(message)
        cuda.memcpy_dtoh(res, io_message_gpu)

        # Record execution time (including memory transfers)
        end.record()
        end.synchronize()

        # return a tuple of output of sine computation and time taken to execute the operation (in ms).
        return res, start.time_till(end) * 10 ** (-3)


def test1_ShiftRowsTest():
    # Input array
    hex_in = "63cab7040953d051cd60e0e7ba70e18c"
    byte_in = bytes.fromhex(hex_in)
    byte_array_in = np.frombuffer(byte_in, dtype=np.byte)

    # Reference output
    hex_ref = "6353e08c0960e104cd70b751bacad0e7"
    byte_ref = bytes.fromhex(hex_ref)
    byte_array_ref = np.frombuffer(byte_ref, dtype=np.byte)

    graphicscomputer = ShiftRowsTest()
    result_gpu = graphicscomputer.shiftrows_gpu(byte_array_in, byte_array_in.size)[0]
    assert np.array_equal(result_gpu, byte_array_ref)

def test2_ShiftRowsTest():
    # Input array
    hex_in = "63cab7040953d051cd60e0e7ba70e18c63cab7040953d051cd60e0e7ba70e18c"
    byte_in = bytes.fromhex(hex_in)
    byte_array_in = np.frombuffer(byte_in, dtype=np.byte)

    # Reference output
    hex_ref = "6353e08c0960e104cd70b751bacad0e76353e08c0960e104cd70b751bacad0e7"
    byte_ref = bytes.fromhex(hex_ref)
    byte_array_ref = np.frombuffer(byte_ref, dtype=np.byte)

    graphicscomputer = ShiftRowsTest()
    result_gpu = graphicscomputer.shiftrows_gpu(byte_array_in, byte_array_in.size)[0]
    assert np.array_equal(result_gpu, byte_array_ref)

