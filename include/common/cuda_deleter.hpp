#pragma once

#include "cuda_helpers.cuh"

template <typename T>
struct CudaDeleter {
    void operator()(T* p) { CUDA_TRY(cudaFree(p)); }
};