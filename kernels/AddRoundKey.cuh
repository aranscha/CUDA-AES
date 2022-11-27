/*
   AddRoundKey operation:
   Apply RoundKey to the block by performing a bitwise EXOR between the bytes
   from the block and the corresponding bytes from the RoundKey.

   Input:
    - RoundKey: char array of length 16
   InOut:
    - block: char array of length 16
*/

__device__ void AddRoundKey(char* block, char* RoundKey)
{
    block[0]  ^= RoundKey[0];
    block[1]  ^= RoundKey[1];
    block[2]  ^= RoundKey[2];
    block[3]  ^= RoundKey[3];
    block[4]  ^= RoundKey[4];
    block[5]  ^= RoundKey[5];
    block[6]  ^= RoundKey[6];
    block[7]  ^= RoundKey[7];
    block[8]  ^= RoundKey[8];
    block[9]  ^= RoundKey[9];
    block[10] ^= RoundKey[10];
    block[11] ^= RoundKey[11];
    block[12] ^= RoundKey[12];
    block[13] ^= RoundKey[13];
    block[14] ^= RoundKey[14];
    block[15] ^= RoundKey[15];
}

__global__ void AddRoundKeyTest(char* message, char* roundkey, const unsigned int length)
{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 16; // * 16 because every thread processes an entire block
    if (idx + 16 <= length)
        AddRoundKey(message + idx, roundkey);
}
