#include "key_generator.h"

__device__ void generate_key(uint64_t key_idx, uint8_t* priv_key) {
    // Simple key generation (replace with real logic if needed)
    memcpy(priv_key, &key_idx, 8);
    memset(priv_key + 8, 0, 24);
}