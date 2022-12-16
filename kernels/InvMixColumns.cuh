// LUTs for the tests are defined in mixColumns. This file should be compiled first, as invMixColumns is dependent on mixColumns

#ifndef LUT_IN_SHARED
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

    #ifndef LUT_IN_SHARED
    mixColumns(block);
    #else
    mixColumns(block, mul2, mul3);
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
