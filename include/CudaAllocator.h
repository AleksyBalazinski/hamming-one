#pragma once

#include <new>

template <typename T>
class CudaAllocator
{
public:
	__host__ T *allocate(size_t n)
	{
		T *p;
		cudaError_t error = cudaMalloc(&p, n * sizeof(T));
		error = cudaMemset(p, 0, n * sizeof(T));
		if (error != cudaSuccess)
		{
			throw std::bad_alloc();
		}

		return p;
	}

	__host__ void deallocate(T *p, size_t) { cudaFree(p); }

	__host__ void construct(T *p)
	{
		// no-op: once memory has been allocated by cudaMalloc(), we cannot access it from host
	}

	__host__ void destroy(T *p)
	{
		// no-op: once memory has been allocated by cudaMalloc(), we cannot access it from host
	}
};