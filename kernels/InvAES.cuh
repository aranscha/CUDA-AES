/*
    The stucture of encryption and decryption is exactly the same, so the fastest 
    encryption method is used to implement the decryption algorithm. The fastest
    method was found to be storing the state in private memory, and the LUTs in shared
    memory, which avoids the use of constant memory

    inputs (different from AES):
        - mem_initialized: indicates whether the LUTs were already loaded 
*/

__global__ void inv_AES(char* state, char* cipherKey, const unsigned int stateLength, char* grcon, char* gsbox, char* ginvsbox, char* gmul2, char* gmul3){
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Load the lookup tables into shared memory
    __shared__ char rcon[256];
    __shared__ char sbox[256]; // required for key expansion
    __shared__ char invsbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                invsbox[i] = ginvsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            invsbox[threadIdx.x] = ginvsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }
    __syncthreads();


    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(cipherKey, ExpandedKey, rcon, sbox);

    // Load State into private memory (a state is only used by a single thread)
    char stateLocal[16];
    if(index + 16 <= stateLength){
        for(int i = 0; i < 16; i++){
            stateLocal[i] = state[index + i];
        }
    }

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= stateLength)
    {
        AddRoundKey(stateLocal, ExpandedKey + 16 * NR_ROUNDS);
        for (int i = 1; i < NR_ROUNDS; i++){
            InvRound(stateLocal, ExpandedKey + 16*NR_ROUNDS - 16*i, invsbox, mul2, mul3); // now run through the expanded key in reverse!
        }
        InvFinalRound(stateLocal, ExpandedKey, invsbox);
    }

    __syncthreads();
    // Write back the results to State
    if (index + 16 <= stateLength)
        for (int i = 0; i < 16; i++){
            state[index + i] = stateLocal[i];
        }

}
