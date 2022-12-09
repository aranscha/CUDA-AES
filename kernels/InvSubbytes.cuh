// analogous to subBytes, but with the inverse of the original sbox
#ifndef LUT_IN_SHARED
__device__ void InvSubBytes(char* block){
#else
__device__ void InvSubBytes(char* block, char* invSbox){
#endif
    for(unsigned int i = 0; i < 16; i++){
        block[i] = invSbox[(unsigned char) block[i]];
    }
}