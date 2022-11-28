#define NR_ROUNDS 10

__constant__ char rcon[256];
__constant__ char sbox[256];

/*
   RotByte:
   Rotate the bytes within a 4-byte array as follows:
   a b c d -> b c d a

   InOut:
    - array: char array of length 4
*/

__device__ void RotByte(char* array)
{
    char temp = array[0];
    array[0] = array[1];
    array[1] = array[2];
    array[2] = array[3];
    array[3] = temp;
}

/*
   SubByte:
   Apply the sbox lookup table to each byte.

   InOut:
    - array: char array of length 4
*/

__device__ void SubByte(char* array)
{
    // Need to convert to unsigned char for correct indexing
    array[0] = sbox[(unsigned char) array[0]];
    array[1] = sbox[(unsigned char) array[1]];
    array[2] = sbox[(unsigned char) array[2]];
    array[3] = sbox[(unsigned char) array[3]];
}

/*
   KeyExpansion:
   Create the expanded key from the Cipher Key. The expanded key has
   16 * (NR_ROUNDS + 1) bytes, and is chopped into pieces of 16 bytes to obtain
   the RoundKey for each round.

   Input:
    - CipherKey: char array of length 4

   Output:
    - ExpandedKey: char array of length 16 * (NR_ROUNDS + 1)
*/

__device__ void KeyExpansion(char* CipherKey, char* ExpandedKey)
{
    // First part of the expanded key is equal to the Cipher key.
    for (int i = 0; i < 16; i++)
        ExpandedKey[i] = CipherKey[i];

    // Obtain the following parts of the ExpandedKey, creating a word (4 bytes)
    // during each iteration
    for (int i = 16; i < 16 * (NR_ROUNDS + 1); i += 4)
    {
        // Store the current last word of the ExpandedKey
        char temp[4];
        for (int j = 0; j < 4; j++)
            temp[j] = ExpandedKey[(i - 4) + j];

        // If the current word is a multiple of the key length, then apply
        // a transformation
        if (i % 16 == 0)
        {
            RotByte(temp);
            SubByte(temp);
            temp[0] ^= rcon[i / 16];
        }

        // The next word of the ExpandedKey is equal to the bitwise EXOR
        // of the current last word and the word came 4 words before the
        // word that is currently computed
        for (int j = 0; j < 4; j++)
            ExpandedKey[i + j] = ExpandedKey[(i - 16) + j] ^ temp[j];
    }

}

__global__ void KeyExpansionTest(char* cipherkey, char* expandedkey)
{
    KeyExpansion(cipherkey, expandedkey);
}
