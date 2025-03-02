#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include <cuda_runtime.h>

struct secp256k1_pubkey {
    uint8_t data[64];
};

__device__ void secp256k1_ec_pubkey_create(const uint8_t* priv_key, secp256k1_pubkey* pubkey);
__device__ void secp256k1_pubkey_serialize(const secp256k1_pubkey* pubkey, uint8_t* output);

#endif