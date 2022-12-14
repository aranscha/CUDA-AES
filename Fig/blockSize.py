import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import tb.AES_test as AES
from tqdm import tqdm
from ref.AES_Python import AES_Python

if __name__ == '__main__':
    graphicscomputer = AES.AESTest()
    aes_cpu = AES_Python()

    block_sizes = [16, 32, 64, 128, 256, 512, 1024]
    test_size = 1073741824

    nr_iterations = 5

    times_gpu_naive = []
    times_gpu_shared = []
    times_gpu_shared_coalesced = []
    times_gpu_shared_coalesced_noconst = []
    times_private = []
    times_private_sharedlut = []
    times_cpu = []

    for block_size in block_sizes:
        with open(f"../tb/test_cases/test_case_{test_size}.txt", "r") as file:
            hex_in = file.read()

        byte_in = bytes.fromhex(hex_in)
        byte_array_in = np.frombuffer(byte_in, dtype=np.byte)

        # Get test key
        hex_key = "000102030405060708090a0b0c0d0e0f"
        byte_key = bytes.fromhex(hex_key)
        byte_array_key = np.frombuffer(byte_key, dtype=np.byte)

        times_gpu_naive_it = []
        times_gpu_shared_it = []
        times_gpu_shared_coalesced_it = []
        times_gpu_shared_coalesced_noconst_it = []
        times_cpu_it = []
        times_private_sharedlut_it = []
        times_private_it = []

        for iteration in tqdm(range(nr_iterations)):
            time_gpu_naive = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "naive", block_size=block_size)[1]
            time_gpu_shared = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "shared", block_size=block_size)[1]
            time_gpu_shared_coalesced = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "shared_coalesced", block_size=block_size)[1]
            time_gpu_shared_coalesced_noconst = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "shared_coalesced_noconst", block_size=block_size)[1]
            time_private = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "private", block_size=block_size)[1]
            time_private_sharedlut = graphicscomputer.AES_gpu(byte_array_in, byte_array_key, byte_array_in.size, "private_sharedlut", block_size=block_size)[1]
            time_cpu = aes_cpu.encrypt(hex_in, hex_key)[1]

            times_gpu_naive_it.append(time_gpu_naive)
            times_gpu_shared_it.append(time_gpu_shared)
            times_gpu_shared_coalesced_it.append(time_gpu_shared_coalesced)
            times_gpu_shared_coalesced_noconst_it.append(time_gpu_shared_coalesced_noconst)
            times_private_it.append(time_private)
            times_private_sharedlut_it.append(time_private_sharedlut)
            times_cpu_it.append(time_cpu)
        
        times_gpu_naive.append(sum(times_gpu_naive_it)/len(times_gpu_naive_it))
        times_gpu_shared.append(sum(times_gpu_shared_it)/len(times_gpu_shared_it))
        times_gpu_shared_coalesced.append(sum(times_gpu_shared_coalesced_it)/len(times_gpu_shared_coalesced_it))
        times_gpu_shared_coalesced_noconst.append(sum(times_gpu_shared_coalesced_noconst_it)/len(times_gpu_shared_coalesced_noconst_it))
        times_private.append(sum(times_private_it)/len(times_private_it))
        times_private_sharedlut.append(sum(times_private_sharedlut_it)/len(times_private_sharedlut_it))
        times_cpu.append(sum(times_cpu_it)/len(times_cpu_it))

    print('GPU (naive) execution times:\n', times_gpu_naive)
    print('GPU (shared) execution times:\n', times_gpu_shared)
    print('GPU (shared & coalesced) execution times:\n', times_gpu_shared_coalesced)
    print('GPU (shared & coalesced, no constant mem) execution times:\n', times_gpu_shared_coalesced_noconst)
    print('GPU (state in private memory) execution times:\n', times_private)
    print('GPU (state in private mem, luts in shared) execution times: \n', times_private_sharedlut)
    print('CPU execution times:\n', times_cpu)
