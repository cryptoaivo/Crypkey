#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

void launch_cuda_kernel(uint64_t start_key, uint64_t batch_size, int gpu_count);
bool check_results(uint8_t* private_key);
uint64_t get_progress();
void reset_counters();

#endif