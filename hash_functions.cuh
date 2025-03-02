#ifndef HASH_FUNCTIONS_CUH
#define HASH_FUNCTIONS_CUH

#include <cuda_runtime.h>

__device__ void hash160_rmdsha(const uint8_t* data, size_t len, uint8_t* out);

#endif