#include "bloom_manager.h"
#include "address_matcher.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <array>

constexpr int BLOOM_FILTER_BITS = 16384;
constexpr std::array<uint32_t, 4> BLOOM_HASH_SEEDS = {0x15a4db35, 0x76a54d32, 0xabcdef12, 0x98765432};
constexpr int ADDRESS_LENGTH = 35;

class BloomManager {
private:
    std::string last_target;
    std::array<uint32_t, BLOOM_FILTER_BITS / 32> bloom_cache;
    cudaStream_t stream;

public:
    BloomManager() { cudaStreamCreate(&stream); }
    ~BloomManager() { cudaStreamDestroy(stream); }

    void init(const std::string& target) {
        if (target.length() != ADDRESS_LENGTH) {
            throw std::invalid_argument("Invalid target address length");
        }
        if (target == last_target) return;

        std::array<uint32_t, BLOOM_FILTER_BITS / 32> bloom_filter = {};
        for (uint32_t seed : BLOOM_HASH_SEEDS) {
            uint32_t h = seed;
            for (char c : target) {
                h = (h ^ static_cast<uint8_t>(c)) * 0x01000193; // Faster FNV variant
            }
            uint32_t pos = h % BLOOM_FILTER_BITS;
            bloom_filter[pos / 32] |= (1U << (pos % 32));
        }

        cudaError_t err = cudaMemcpyToSymbolAsync(
            AddressMatcher::bloom_filter, bloom_filter.data(), BLOOM_FILTER_BITS / 8,
            0, cudaMemcpyHostToDevice, stream
        );
        if (err != cudaSuccess) {
            throw std::runtime_error("Bloom filter copy failed: " + std::string(cudaGetErrorString(err)));
        }

        bloom_cache = bloom_filter;
        last_target = target;
    }

    void precompute_puzzles() {
        std::vector<std::string> targets = {
            "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", // #68
            "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", // #69
            "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"  // #71
        };
        std::array<uint32_t, BLOOM_FILTER_BITS / 32> bloom_filter = {};
        for (const auto& target : targets) {
            for (uint32_t seed : BLOOM_HASH_SEEDS) {
                uint32_t h = seed;
                for (char c : target) {
                    h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
                }
                uint32_t pos = h % BLOOM_FILTER_BITS;
                bloom_filter[pos / 32] |= (1U << (pos % 32));
            }
        }
        cudaMemcpyToSymbolAsync(
            AddressMatcher::bloom_filter, bloom_filter.data(), BLOOM_FILTER_BITS / 8,
            0, cudaMemcpyHostToDevice, stream
        );
        bloom_cache = bloom_filter;
        last_target = "puzzles";
    }

    void reset() {
        std::array<uint32_t, BLOOM_FILTER_BITS / 32> empty_filter = {};
        cudaMemcpyToSymbol(
            AddressMatcher::bloom_filter, empty_filter.data(), BLOOM_FILTER_BITS / 8
        );
        bloom_cache.fill(0);
        last_target.clear();
    }
};