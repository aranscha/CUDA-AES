/*
   ShiftRows operation:
   Shift the bytes within a block as follows:
   b0  b4  b8  b12 -> b0  b4  b8  b12 (no shift)
   b1  b5  b9  b13 -> b5  b9  b13 b1  (shift left by one)
   b2  b6  b10 b14 -> b10 b14 b2  b6  (shift left by two)
   b3  b7  b11 b15 -> b15 b3  b7  b11 (shift left by three)

   InOut:
    - block: char array of length 16
*/

__device__ void ShiftRows(char* block)
{
    // ROW 0: Untouched

    // ROW 1: Shift left by one
    char temp = block[1];
    block[1]  = block[5];
    block[5]  = block[9];
    block[9]  = block[13];
    block[13] = temp;

    // ROW 2: Shift left by two
    temp = block[2];
    block[2]  = block[10];
    block[10] = temp;
    temp = block[6];
    block[6]  = block[14];
    block[14] = temp;

    // ROW 3: Shift left by three
    temp = block[15];
    block[15] = block[11];
    block[11] = block[7];
    block[7]  = block[3];
    block[3]  = temp;
}

__global__ void ShiftRowsTest(char* message, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block
    if (idx + 16 < length)
        ShiftRows(message + idx);
}
