#pragma once

#include <cuda/atomic>
#include <cuda/std/atomic>

class CudaLock
{
    cuda::std::atomic_flag mutex_;

public:
    __device__ void lock()
    {
        while (mutex_.test_and_set(cuda::memory_order_acquire))
            ;
    }
    __device__ void unlock()
    {
        mutex_.clear(cuda::memory_order_release);
    }
};