#include "cuda_kernels.h"
#include "address_matcher.h"
#include "key_generator.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Configuration
constexpr int THREADS_PER_BLOCK = 1024;
constexpr int KEYS_PER_THREAD = 32;
constexpr int UNROLL_FACTOR = 8;

// Global counters (managed memory)
__device__ __managed__ uint64_t global_key_counter = 0;
__device__ __managed__ bool key_found_flag = false;
__device__ __managed__ uint8_t found_private_key[32];

#include "secp256k1.cuh"
#include "hash_functions.cuh"

// Device Function: Generate Address
__device__ void generate_address(const uint8_t* priv_key, uint8_t* hash160) {
    secp256k1_pubkey pubkey;
    secp256k1_ec_pubkey_create(priv_key, &pubkey);
    uint8_t pub_serialized[33];
    secp256k1_pubkey_serialize(&pubkey, pub_serialized);
    hash160_rmdsha(pub_serialized, 33, hash160);
}

// CUDA Kernel
__global__ void crack_kernel(uint64_t start_key, uint64_t total_keys, int gpu_id, int gpu_count) {
    uint64_t thread_start = start_key + 
        (blockIdx.x * blockDim.x + threadIdx.x) * KEYS_PER_THREAD +
        (gpu_id * total_keys / gpu_count);
    
    __shared__ uint64_t block_counter;
    __shared__ uint8_t shared_hash[THREADS_PER_BLOCK][20];
    uint8_t priv_key[32] = {0};
    uint8_t* hash160 = shared_hash[threadIdx.x];
    uint64_t local_count = 0;

    if (threadIdx.x == 0) block_counter = 0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; i += UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            uint64_t current_key = thread_start + i * blockDim.x + u;
            if (current_key >= start_key + total_keys / gpu_count) break;
            memcpy(priv_key, ¤t_key, 8);
            memset(priv_key + 8, 0, 24);
            generate_address(priv_key, hash160);
            if (AddressMatcher::bloomCheck(hash160) && AddressMatcher::compare(hash160)) {
                if (atomicCAS(&key_found_flag, 0, 1) == 0) {
                    memcpy(found_private_key, priv_key, 32);
                }
            }
            local_count++;
        }
    }

    atomicAdd(&block_counter, local_count);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&global_key_counter, block_counter);
}

// Host Functions
void launch_cuda_kernel(uint64_t start_key, uint64_t batch_size, int gpu_count) {
    int devices;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess) {
        Logger::log("CUDA error: Failed to get device count - " + std::string(cudaGetErrorString(err)));
        return;
    }
    gpu_count = std::min(gpu_count, devices);
    for (int i = 0; i < gpu_count; i++) {
        cudaSetDevice(i);
        dim3 blocks((batch_size / gpu_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 threads(THREADS_PER_BLOCK);
        crack_kernel<<<blocks, threads>>>(start_key, batch_size, i, gpu_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::log("CUDA kernel launch failed on GPU " + std::to_string(i) + ": " + cudaGetErrorString(err));
        }
    }
    for (int i = 0; i < gpu_count; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::log("CUDA sync failed on GPU " + std::to_string(i) + ": " + cudaGetErrorString(err));
        }
    }
}

bool check_results(uint8_t* private_key) {
    bool found;
    cudaMemcpy(&found, &key_found_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    if (found) {
        cudaMemcpy(private_key, found_private_key, 32, cudaMemcpyDeviceToHost);
        return true;
    }
    return false;
}

uint64_t get_progress() {
    uint64_t progress;
    cudaMemcpy(&progress, &global_key_counter, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return progress;
}

void reset_counters() {
    cudaMemset(&global_key_counter, 0, sizeof(uint64_t));
    cudaMemset(&key_found_flag, 0, sizeof(bool));
    cudaMemset(found_private_key, 0, 32);
}