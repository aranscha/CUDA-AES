#!/usr/bin/env python

import time
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class AddRoundKeyTest:
    def __init__(self):
        self.getSourceModule()

    def getSourceModule(self):
        file = open("../kernels/AddRoundKey.cuh", "r")
        kernelwrapper = file.read()
        file.close()
        self.module = SourceModule(kernelwrapper)


    def addroundkey_gpu(self, message, roundkey, length):
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()

        # Start recording execution time
        start.record()

        # Device memory allocation for input and output arrays
        io_message_gpu = cuda.mem_alloc_like(message)
        i_roundkey_gpu = cuda.mem_alloc_like(roundkey)

        # Copy data from host to device
        cuda.memcpy_htod(io_message_gpu, message)
        cuda.memcpy_htod(io_roundkey_gpu, roundkey)

        # Call the kernel function from the compiled module
        prg = self.module.get_function("AddRoundKeyTest")

        # Calculate block size and grid size
        block_size = length
        grid_size = 1
        if (block_size > 1024):
            block_size = 1024
            grid_size = (length - 1) / 1024 + 1;

        blockDim = (block_size, 1, 1)
        gridDim = (grid_size, 1, 1)

        # Call the kernel loaded to the device
        prg(io_message_gpu, i_roundkey_gpu, np.uint32(length), block=blockDim, grid=gridDim)

        # Copy result from device to the host
        res = np.empty_like(message)
        cuda.memcpy_dtoh(res, io_message_gpu)

        # Record execution time (including memory transfers)
        end.record()
        end.synchronize()

        # return a tuple of output of sine computation and time taken to execute the operation (in ms).
        return res, start.time_till(end) * 10 ** (-3)


def test1_AddRoundKeyTest():
    # Input arrays
    hex_in1 = "5f72641557f5bc92f7be3b291db9f91a"
    byte_in1 = bytes.fromhex(hex_in1)
    byte_array_in1 = np.frombuffer(byte_in1, dtype=np.byte)

    hex_in2 = "000102030405060708090a0b0c0d0e0f"
    byte_in2 = bytes.fromhex(hex_in2)
    byte_array_in2 = np.frombuffer(byte_in2, dtype=np.byte)

    # Reference output
    hex_ref = "5f73661653f0ba95ffb7312211b4f715"
    byte_ref = bytes.fromhex(hex_ref)
    byte_array_ref = np.frombuffer(byte_ref, dtype=np.byte)

    graphicscomputer = ShiftRowsTest()
    result_gpu = graphicscomputer.shiftrows_gpu(byte_array_in1, byte_array_in2, byte_array_in.size)[0]
    assert np.array_equal(result_gpu, byte_array_ref)
