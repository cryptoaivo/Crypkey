#include "address_matcher.h"
#include <cuda_runtime.h>
#include <iostream>

// Constants
constexpr int BLOOM_HASH_SEEDS[] = {0x15a4db35, 0x76a54d32, 0xabcdef12, 0x98765432};
constexpr int NUM_HASH_SEEDS = 4;
constexpr int ADDRESS_LENGTH = 35;

// Device Constants
__constant__ uint32_t AddressMatcher::bloom_filter[BLOOM_FILTER_BITS / 32];
__constant__ uint8_t AddressMatcher::target_hash160[HASH160_LENGTH];

// Host Initialization
void AddressMatcher::init(const char* target) {
    if (!target) {
        throw std::invalid_argument("Null target address in AddressMatcher::init().");
    }

    // Placeholder: Decode Base58 to hash160 (needs real implementation)
    uint8_t host_hash160[HASH160_LENGTH] = {0};
    // TODO: Add Base58Check decoding (e.g., using a library or custom code)
    for (int i = 0; i < HASH160_LENGTH; i++) host_hash160[i] = target[i % HASH160_LENGTH]; // Dummy

    cudaError_t err = cudaMemcpyToSymbol(
        target_hash160, host_hash160, HASH160_LENGTH, 0, cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        Logger::log("Failed to copy target_hash160: " + std::string(cudaGetErrorString(err)));
        throw std::runtime_error("CUDA error");
    }

    uint32_t host_bloom[BLOOM_FILTER_BITS / 32] = {0};
    for (int seed : BLOOM_HASH_SEEDS) {
        uint32_t h = seed;
        for (int i = 0; i < ADDRESS_LENGTH; i++) {
            h = (h * 0x01000193) ^ static_cast<uint8_t>(target[i]);
        }
        h %= BLOOM_FILTER_BITS;
        host_bloom[h / 32] |= (1 << (h % 32));
    }

    err = cudaMemcpyToSymbol(
        bloom_filter, host_bloom, sizeof(host_bloom), 0, cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        Logger::log("Failed to copy bloom_filter: " + std::string(cudaGetErrorString(err)));
        throw std::runtime_error("CUDA error");
    }
}

// Device Functions
__device__ bool AddressMatcher::compare(const uint8_t* candidate_hash160) {
    const uint4* a4 = reinterpret_cast<const uint4*>(candidate_hash160);
    const uint4* b4 = reinterpret_cast<const uint4*>(target_hash160);
    bool match = (a4[0].x == b4[0].x) && (a4[0].y == b4[0].y) &&
                 (a4[0].z == b4[0].z) && (a4[0].w == b4[0].w);
    return match && (__ldg(&candidate_hash160[16]) == __ldg(&target_hash160[16])) &&
                   (__ldg(&candidate_hash160[17]) == __ldg(&target_hash160[17])) &&
                   (__ldg(&candidate_hash160[18]) == __ldg(&target_hash160[18])) &&
                   (__ldg(&candidate_hash160[19]) == __ldg(&target_hash160[19]));
}

__device__ bool AddressMatcher::bloomCheck(const uint8_t* hash160) {
    #pragma unroll
    for (int seed : BLOOM_HASH_SEEDS) {
        uint32_t h = hash(hash160, HASH160_LENGTH, seed);
        if (!(bloom_filter[h / 32] & (1 << (h % 32)))) return false;
    }
    return true;
}

__device__ uint32_t AddressMatcher::hash(const uint8_t* data, int len, int seed) {
    uint32_t h = seed;
    #pragma unroll
    for (int i = 0; i < len; i++) {
        h = (h ^ static_cast<uint8_t>(data[i])) * 0x01000193;
    }
    return h % BLOOM_FILTER_BITS;
}