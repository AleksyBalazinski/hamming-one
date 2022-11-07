#pragma once

class CudaLock
{
	int *mutex;

public:
	CudaLock()
	{
		int state = 0;
		cudaMalloc((void **)&mutex, sizeof(int));
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
	}
	__device__ void lock()
	{
		while (atomicCAS(mutex, 0, 1) != 0)
			;
		__threadfence();
	}
	__device__ void unlock()
	{
		atomicExch(mutex, 0);
		__threadfence();
	}
};