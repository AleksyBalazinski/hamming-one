#pragma once

#include <functional>

template <typename T, typename Allocator = std::allocator<T>>
class Array
{
private:
    T *storage = nullptr;
    size_t N;
    Allocator allocator;

    template <typename U, typename Allocator>
    friend class Array;

public:
    __host__ Array() : N(0) {}

    __host__ Array(size_t N) : N(N)
    {
        storage = allocator.allocate(N);
        for (size_t i = 0; i < N; i++)
            allocator.construct(storage + i);
    }

    __host__ __device__ T operator[](size_t i) const { return storage[i]; }

    __host__ __device__ T &operator[](size_t i) { return storage[i]; }

    __host__ __device__ size_t size() { return N; }

    // copies all underlying memory to DST
    template <typename U, typename Allocator>
    __host__ cudaError_t copyTo(Array<U, Allocator> &dst, cudaMemcpyKind kind)
    {
        cudaError_t error = cudaMemcpy(dst.storage, this->storage, N * sizeof(T), kind);

        return error;
    }

    __host__ ~Array()
    {
        for (size_t i = 0; i < N; i++)
            allocator.destroy(storage + i);
        allocator.deallocate(storage, N);
    }
};