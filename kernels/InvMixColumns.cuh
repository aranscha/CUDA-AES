#ifdef TEST_INVMIXCOLUMNS

__constant__ char mul2[256];
__constant__ char mul3[256];

# endif

#ifndef AES_SHARED_COALESCED_NOCONST
__device__ void mixColumns(char* block){
#else
__device__ void mixColumns(char* block, char* mul2, char* mul3){
#endif
    char temp[16];
    for (unsigned int col = 0; col < 4; col++){
        temp[4*col] = mul2[(unsigned char) block[4*col]] ^ mul3[(unsigned char) block[4*col + 1]] ^ block[4*col + 2] ^ block[4*col + 3];
        temp[4*col + 1] = block[4*col] ^ mul2[(unsigned char) block[4*col + 1]] ^ mul3[(unsigned char) block[4*col + 2]] ^ block[4*col + 3];
        temp[4*col + 2] = block[4*col] ^ block[4*col + 1] ^ mul2[(unsigned char) block[4*col + 2]] ^ mul3[(unsigned char) block[4*col + 3]];
        temp[4*col + 3] = mul3[(unsigned char) block[4*col]] ^ block[4*col + 1] ^ block[4*col + 2] ^ mul2[(unsigned char) block[4*col + 3]];

    }

    for (unsigned int i = 0; i < 16; i++){
        block[i] = temp[i];
    }
}

#ifndef AES_SHARED_COALESCED_NOCONST
__device__ void invMixColumns(char* block){
#else
__device__ void invMixColumns(char* block, char* mul2, char* mul3){
#endif
    char u;
    char v;
    for (unsigned int col = 0; col < 4; col++){
        u = mul2[(unsigned char) mul2[(unsigned char) (block[4*col] ^ block[4*col + 2])]];
        v = mul2[(unsigned char) mul2[(unsigned char) (block[4*col + 1] ^ block[4*col + 3])]];
        block[4*col] = block[4*col] ^ u;
        block[4*col + 1] = block[4*col + 1] ^ v;
        block[4*col + 2] = block[4*col + 2] ^ u;
        block[4*col + 3] = block[4*col + 3] ^ v;
    }

    #ifndef AES_SHARED_COALESCED_NOCONST
    mixColumns(block);
    #else
    mixColumns(block, mul2, mul3)
    #endif
}


#ifdef TEST_INVMIXCOLUMNS

__global__ void invMixColumnsTest(char* message, const unsigned int length){
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    if (idx + 16 <= length){
        invMixColumns(message + idx);
    }
}

#endif
