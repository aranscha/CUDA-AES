/*
   Round operation:
   Perform the following operations on the input block:
    1. SubBytes
    2. ShiftRows
    3. MixColumns
    4. AddRoundKey
   This operation as a whole is a single round operation from the AES
   algorithm. The RoundKey used is one block of the ExpandedKey.

   Input:
    - roundkey: char array of length 16

   InOut:
    - block: char array of length 16
*/

#ifndef LUT_IN_SHARED
__device__ void Round(char* block, char* roundkey)
{
    SubBytes(block);
    ShiftRows(block);
    mixColumns(block);
    AddRoundKey(block, roundkey);
}
#else
__device__ void Round(char* block, char* roundkey, char* sbox, char* mul2, char* mul3)
{
    SubBytes(block, sbox);
    ShiftRows(block);
    mixColumns(block, mul2, mul3);
    AddRoundKey(block, roundkey);
}
#endif

#ifdef TEST_ROUND

__global__ void RoundTest(char* block, char* roundkey)
{
    Round(block, roundkey);
}

#endif
