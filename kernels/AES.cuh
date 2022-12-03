/*
   AES_naive is a naive implementation of the AES algorithm. It consists of:
    1. KeyExpansion: create the ExpanedKey from the CipherKey
    2. For each block (handled by a seperate thread) apply NR_ROUND - 1 Round
       operations. Each Round consists of:
        a. ByteSub operation
        b. ShiftRows operation
        c. ShiftColumns operation
        d. AddRoundKey operation
    3. For each block apply a single FinalRound operation. FinalRound consists of:
        a. ByteSub operation
        b. ShiftRows operation
        c. AddRoundKey operations

   InOut:
    - State: char array of arbitrary length
   Inputs:
    - CipherKey: char array of length 16
    - StateLength: length of the State array
*/

#ifdef AES_NAIVE

__global__ void AES_naive(char* State, char* CipherKey, const unsigned int StateLength)
{
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey);

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= StateLength)
    {
        AddRoundKey(State + index, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(State + index, ExpandedKey + 16 * i);
        FinalRound(State + index, ExpandedKey + 16 * NR_ROUNDS);
    }
}

#endif

#ifdef AES_SHARED

__global__ void AES_shared(char* State, char* CipherKey, const unsigned int StateLength)
{
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey);

    // Load State into shared memory - not yet coalesced
    __shared__ char StateShared[16*1024];
    int local_index = threadIdx.x * 16;
    if (index + 16 <= StateLength)
	      for (int j = 0; j < 16; j++)
	          StateShared[local_index + j] = State[index + j];

    // Synchronize the threads because thread 0 wrote to shared memory, and
    // the ExpanedKey will be accessed by each thread in the block.
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    if (index + 16 <= StateLength)
    {
        AddRoundKey(StateShared + local_index, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(StateShared + local_index, ExpandedKey + 16 * i);
        FinalRound(StateShared + local_index, ExpandedKey + 16 * NR_ROUNDS);
    }

    // Write back the results to State - not yet coalesced
    if (index + 16 <= StateLength)
        for (int j = 0; j < 16; j++)
            State[index + j] = StateShared[local_index + j];
}

#endif

#ifdef AES_SHARED_COALESCED

__global__ void AES_shared_coalesced(char* State, char* CipherKey, const unsigned int StateLength)
{
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey);

    // Load State into shared memory - coalesced
    __shared__ char StateShared[16*1024];
    if (index + 16 <= StateLength)
      	for (int j = 0; j < 16; j++)
	          StateShared[j * blockDim.x + threadIdx.x] = State[blockIdx.x * blockDim.x * 16 + j * blockDim.x + threadIdx.x];

    // Synchronize the threads
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    int local_index = threadIdx.x * 16;
    if (index + 16 <= StateLength)
    {
        AddRoundKey(StateShared + local_index, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(StateShared + local_index, ExpandedKey + 16 * i);
        FinalRound(StateShared + local_index, ExpandedKey + 16 * NR_ROUNDS);
    }

    // Synchronize the threads
    __syncthreads();

    // Write back the results to State - coalesced
    if (index + 16 <= StateLength)
        for (int j = 0; j < 16; j++)
            State[blockIdx.x * blockDim.x * 16 + j * blockDim.x + threadIdx.x] = StateShared[j * blockDim.x + threadIdx.x];
}

#endif

#ifdef AES_SHARED_COALESCED_NOCONST

__global__ void AES_shared_coalesced_noconst(char* State, char* CipherKey, const unsigned int StateLength, char* grcon, char* gsbox, char* gmul2, char* gmul3)
{
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block

    // Load the lookup tables into shared memory
    __shared__ char rcon[256];
    __shared__ char sbox[256];
    __shared__ char mul2[256];
    __shared__ char mul3[256];

    if (blockDim.x < 256) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 256; i++) {
                rcon[i] = grcon[i];
                sbox[i] = gsbox[i];
                mul2[i] = gmul2[i];
                mul3[i] = gmul3[i];
            }
        }
    } else {
        if (threadIdx.x < 256) {
            rcon[threadIdx.x] = grcon[threadIdx.x];
            sbox[threadIdx.x] = gsbox[threadIdx.x];
            mul2[threadIdx.x] = gmul2[threadIdx.x];
            mul3[threadIdx.x] = gmul3[threadIdx.x];
        }
    }

    // Only a single thread from the thread block must calculate the ExpanedKey
    __shared__ char ExpandedKey[16 * (NR_ROUNDS + 1)];
    if (threadIdx.x == 0)
        KeyExpansion(CipherKey, ExpandedKey, rcon, sbox);

    // Load State into shared memory - coalesced
    __shared__ char StateShared[16*1024];
    if (index + 16 <= StateLength)
      	for (int j = 0; j < 16; j++)
	          StateShared[j * blockDim.x + threadIdx.x] = State[blockIdx.x * blockDim.x * 16 + j * blockDim.x + threadIdx.x];

    // Synchronize the threads
    __syncthreads();

    // Each thread handles 16 bytes (a single block) of the State
    int local_index = threadIdx.x * 16;
    if (index + 16 <= StateLength)
    {
        AddRoundKey(StateShared + local_index, ExpandedKey);
        for (int i = 1; i < NR_ROUNDS; i++)
            Round(StateShared + local_index, ExpandedKey + 16 * i, sbox, mul2, mul3);
        FinalRound(StateShared + local_index, ExpandedKey + 16 * NR_ROUNDS, sbox);
    }

    // Synchronize the threads
    __syncthreads();

    // Write back the results to State - coalesced
    if (index + 16 <= StateLength)
        for (int j = 0; j < 16; j++)
            State[blockIdx.x * blockDim.x * 16 + j * blockDim.x + threadIdx.x] = StateShared[j * blockDim.x + threadIdx.x];
}

#endif
