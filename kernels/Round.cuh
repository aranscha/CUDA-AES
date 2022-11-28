#ifdef TEST_ROUND

#define NR_ROUNDS 10

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif

/*
   Round operation:
   Perform the following operations on the input block:
    1. ByteSub
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
    ByteSub(block);
    ShiftRows(block);
    MixColumns(block);
    AddRoundKey(block, roundkey);
}

#ifdef TEST_ROUND

__global__ void RoundTest(char* block, char* roundkey)
{
    Round(block, roundkey);
}

#endif
