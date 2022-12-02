#ifdef TEST_FINALROUND

#define NR_ROUNDS 10

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif

/*
   FinalRound operation:
   Perform the following operations on the input block:
    1. ByteSub
    2. ShiftRows
    3. AddRoundKey
   This operation as a whole is the final round operation from the AES
   algorithm. The RoundKey used is one block of the ExpandedKey.

   Input:
    - roundkey: char array of length 16

   InOut:
    - block: char array of length 16
*/

__device__ void FinalRound(char* block, char* roundkey)
{
    SubBytes(block);
    ShiftRows(block);
    AddRoundKey(block, roundkey);
}

#ifdef FINALTEST_ROUND

__global__ void FinalRoundTest(char* block, char* roundkey)
{
    FinalRound(block, roundkey);
}

#endif
