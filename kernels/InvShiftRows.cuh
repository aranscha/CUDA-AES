__device__ void InvShiftRows(char* block)
{
    // ROW 0: Untouched

    // ROW 1: Shift right by one
    char temp = block[13];
    block[13] = block[9];
    block[9] = block[5];
    block[5] = block[1];
    block[1] = temp;

    // ROW 2: Shift right by two
    temp = block[14];
    block[14] = block[6];
    block[6] = temp;
    temp = block[10];
    block[10] = block[2];
    block[2] = temp;

    // ROW 3: Shift right by three
    temp = block[3];
    block[3] = block[7];
    block[7] = block[11];
    block[11] = block[15];
    block[15] = temp;

}

#ifdef TEST_INVSHIFTROWS

__global__ void InvShiftRowsTest(char* message, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    if (idx + 16 <= length)
        InvShiftRows(message + idx);
}

#endif
