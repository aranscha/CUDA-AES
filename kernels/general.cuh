#define NR_ROUNDS 10

#ifndef LUT_IN_SHARED

__constant__ char rcon[256];
__constant__ char sbox[256];
__constant__ char invSbox[256];
__constant__ char mul2[256];
__constant__ char mul3[256];

#endif
