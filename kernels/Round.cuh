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

__device__ void Round(char* block, char* roundkey)
{
    SubBytes(block);
    ShiftRows(block);
    mixColumns(block);
    AddRoundKey(block, roundkey);
}

#ifdef TEST_ROUND

__global__ void RoundTest(char* block, char* roundkey)
{
    Round(block, roundkey);
}

#endif
