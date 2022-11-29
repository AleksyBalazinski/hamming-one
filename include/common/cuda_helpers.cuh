#pragma once

#define CUDA_TRY(call)                                                                        \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::terminate();                                                                 \
        }                                                                                     \
    } while (0)