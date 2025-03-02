#ifndef KEY_GENERATOR_H
#define KEY_GENERATOR_H

#include <cuda_runtime.h>

__device__ void generate_key(uint64_t key_idx, uint8_t* priv_key);

#endif