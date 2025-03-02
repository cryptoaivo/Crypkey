#ifndef ADDRESS_MATCHER_H
#define ADDRESS_MATCHER_H

#include <cuda_runtime.h>

namespace AddressMatcher {
    constexpr int BLOOM_FILTER_BITS = 16384;
    constexpr int HASH160_LENGTH = 20;

    // Device constants
    extern __constant__ uint32_t bloom_filter[BLOOM_FILTER_BITS / 32];
    extern __constant__ uint8_t target_hash160[HASH160_LENGTH];

    // Host function
    void init(const char* target);

    // Device functions
    __device__ bool bloomCheck(const uint8_t* hash160);
    __device__ bool compare(const uint8_t* hash160);
    __device__ uint32_t hash(const uint8_t* data, int len, int seed);
}

#endif