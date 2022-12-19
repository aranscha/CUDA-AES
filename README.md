# High-throughput Implementation of the Advanced Encryption Standard Using a GPU
## Abstract
Recently, GPUs have become a go-to solution for accelerating demanding applications. In this work, the authors present a parallel implementation of the Advanced Encryption Standard (AES), using the PyCUDA framework. It was investigated how to map the resources required for this algorithm onto the CUDA memory hierarchy to achieve the best possible performance. The different versions were compared and profiled. For input data larger than 1MB, every version is faster than an existing CPU-based implementation. The best implementation was found to be 6x faster compared to the CPU for large files, and has a throughput of 20.54Gbps. Detailed results and a description of the system configuration can be found in the `E4750_2022Fall_PAES_anr2157_rd3033.report.pdf` file.

## Dependencies
It is recommended to use Python 3.6.9 when running the code.
The packages listed below are required to run the tests, performed experiments, and the `AES.py` script if you want to encode a .txt file. The listed versions are recommended.
```console
cryptography        38.0.4
matplotlib          3.3.4
numpy               1.19.5
pycuda              2019.1.2
pytest              7.0.1
tqdm                4.64.1
```
Note that when running a file that also calls a function from the `cryptography` library, you could get a warning that the version is deprecated. You can simply ignore this however, as it shouldn't affect the results of the code.

## Repository Outline
```bash
.
├── AES.py
├── Fig
│   ├── blockSize.py
│   ├── graphs.ipynb
│   └── inputdataSize.py
├── README.md
├── demo_input.txt
├── demo_output_decrypted.txt
├── demo_output_encrypted.txt
├── kernels
│   ├── AES.cuh
│   ├── AddRoundKey.cuh
│   ├── FinalRound.cuh
│   ├── InvAES.cuh
│   ├── InvFinalRound.cuh
│   ├── InvMixColumns.cuh
│   ├── InvRound.cuh
│   ├── InvShiftRows.cuh
│   ├── InvSubbytes.cuh
│   ├── KeyExpansion.cuh
│   ├── MixColumns.cuh
│   ├── Round.cuh
│   ├── ShiftRows.cuh
│   ├── SubBytes.cuh
│   └── general.cuh
├── profiling_reports
│   └── metrics.nsight-cuprof-report
├── E4750_2022Fall_PAES_anr2157_rd3033.report.pdf
├── ref
│   ├── AES_Python.py
│   ├── __init__.py
└── tb
    ├── AES_test.py
    ├── AES_test.txt
    ├── AES_time.py
    ├── AddRoundKey_test.py
    ├── InvAES_test.py
    ├── InvMixColumns_test.py
    ├── InvRound_test.py
    ├── InvShiftRows_test.py
    ├── KeyExpansion_test.py
    ├── MixColumns_test.py
    ├── Round_test.py
    ├── ShiftRows_test.py
    ├── SubBytes_test.py
    ├── __init__.py
    └── test_cases
        ├── create_test_cases.py
```

## Running automatic tests
Tests were written for all operations, including the inverse operations. These automatic tests can be run by executing the following command in the `\tb` folder (tb stands for testbench).
```console
foo@bar:~/tb$ pytest
```

Note that if you want to run the timing experiments yourself, it is required to run the `create_test_cases.py` file first to generate the random input files.
Running all tests should take about 30s.

## Encrypting and Decrypyting a `.txt` file
You can encrypt and decrypt any `.txt` file by using the `AES.py` script as follows:
```console
foo@bar:~$ python AES.py input.txt output_encrypted.txt output_decrypted.txt
```
The input file should be present in the folder before running the script. The 2 output `.txt` files will then be automatically generated.
