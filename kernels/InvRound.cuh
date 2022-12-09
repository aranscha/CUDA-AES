#ifndef LUT_IN_SHARED
__device__ void InvRound(char* block, char* roundKey){
    InvShiftRows(block);
    InvSubBytes(block);
    AddRoundKey(block, roundKey);
    invMixColumns(block);
}
#else
__device__ void InvRound(char* block, char* roundKey, char* invSbox, char* mul2, char* mul3){
    InvShiftRows(block);
    InvSubBytes(block, invSbox);
    AddRoundKey(block, roundKey);
    invMixColumns(block, mul2, mul3);
}
#endif

#ifdef TEST_INVROUND

__global__ void InvRoundTest(char* block, char* roundkey)
{
    InvRound(block, roundkey);
}

#endif