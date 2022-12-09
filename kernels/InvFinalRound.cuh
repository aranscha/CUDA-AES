#ifndef LUT_IN_SHARED
__device__ void InvFinalRound(char* block, char* roundkey)
{
    InvShiftRows(block);
    InvSubBytes(block);
    AddRoundKey(block, roundkey);
}
#else
__device__ void InvFinalRound(char* block, char* roundkey, char* invSbox)
{
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundkey);
}
#endif
