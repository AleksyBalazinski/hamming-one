#pragma once

#include "detail/cuda_helpers.cuh"

template <typename T>
struct cuda_deleter
{
	void operator()(T *p) { CUDA_TRY(cudaFree(p)); }
};

template <typename T>
class CudaAllocator
{
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef T &reference;
	typedef const T &const_reference;

	template <class U>
	struct rebind
	{
		typedef CudaAllocator<U> other;
	};
	CudaAllocator() = default;
	template <class U>
	constexpr CudaAllocator(const CudaAllocator<U> &) noexcept {}

	T *allocate(size_t n)
	{
		void *p = nullptr;
		CUDA_TRY(cudaMalloc(&p, n * sizeof(T)));

		return static_cast<T *>(p);
	}

	void
	deallocate(T *p, size_t)
	{
		CUDA_TRY(cudaFree(p));
	}

	void construct(T *p)
	{
		// no-op: once memory has been allocated by cudaMalloc(), we cannot access it from host
	}

	void destroy(T *p)
	{
		// no-op: once memory has been allocated by cudaMalloc(), we cannot access it from host
	}
};