#pragma once

class CudaLock
{
	int *mutex;

public:
	CudaLock()
	{
		cudaMalloc((void **)&mutex, sizeof(int));
		cudaMemset(mutex, 0, sizeof(int));
	}
	__device__ void lock()
	{
		while (atomicCAS(mutex, 0, 1) != 0)
			;
		__threadfence();
	}
	__device__ void unlock()
	{
		__threadfence();
		atomicExch(mutex, 0);
	}

	~CudaLock()
	{
		cudaFree(mutex);
	}
};