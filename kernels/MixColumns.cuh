#ifdef TEST_MIXCOLUMNS

__constant__ char mul3[256]; // look-up table for the multiplication of a polynomial with x + 1

# endif
/*
    xtime:
        There is no simple operation on the byte level that corresponds to a
        finite field multiplication. A multiplication by x, however, can be implemented
        as a byte level operation. Thus, multiplication by higher powers of x are 
        repetitions of xtime.

        input:
            poly: byte representing a polynomial 
        output:
            out: poly multiplied by x 
*/

__device__ char xtime(char poly){
    char out = poly << 1;
    if (!(out & 0x80)){     // polynomial not yet in reduced form
        out ^= 0x1b;
    }
    return out;
}

/* Multiply input polynomial by x + 1
__device__ char mul3(char poly){
    return poly ^ xtime(poly);
}
*/

__device__ void mixColumns(char* block){
    char temp[16]; 
    // char t;
    for (unsigned int col = 0; col < 4; col++){
        // t = (char) block[4*col] ^ block[4*col + 1] ^ block[4*col+ 2] ^ block[4*col + 3];
        // temp[4*col] = (char) block[4*col] ^ xtime(block[4*col] ^ block[4*col + 1]) ^ t;
        // temp[4*col + 1] = (char) block[4*col + 1] ^ xtime(block[4*col + 1] ^ block[4*col + 2]) ^ t;  
        // temp[4*col + 2] = (char) block[4*col + 2] ^ xtime(block[4*col + 2] ^ block[4*col + 3]) ^ t;  
        // temp[4*col + 3] = (char) block[4*col + 3] ^ xtime(block[4*col + 3] ^ block[4*col]) ^ t;  
        temp[4*col] = xtime(block[4*col]) ^ mul3[(unsigned char) block[4*col + 1]] ^ block[4*col + 2] ^ block[4*col + 3];
        temp[4*col + 1] = block[4*col] ^ xtime(block[4*col + 1]) ^ mul3[(unsigned char) block[4*col + 2]] ^ block[4*col + 3];
        temp[4*col + 2] = block[4*col] ^ block[4*col + 1] ^ xtime(block[4*col + 2]) ^ mul3[(unsigned char) block[4*col + 3]];
        temp[4*col + 3] = mul3[(unsigned char) block[4*col]] ^ block[4*col + 1] ^ block[4*col + 2] ^ xtime(block[4*col + 3]);
    }
    
    for (unsigned int i = 0; i < 16; i++){
        block[i] = temp[i];
    }
}

#ifdef TEST_MIXCOLUMNS

__global__ void xtime_test(char* poly){
    if(threadIdx.x == 0){
        poly[blockIdx.x] = xtime(poly[blockIdx.x]);
    }
}

__global__ void mixColumnsTest(char* message, const unsigned int length){
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16;

    if (idx + 16 <= length){
        mixColumns(message + idx);
    }
}

#endif 