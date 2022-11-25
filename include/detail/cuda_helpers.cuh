#pragma once

#define CUDA_TRY(call)                                                                        \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess)                                                               \
        {                                                                                     \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::terminate();                                                                 \
        }                                                                                     \
    } while (0)

void set_device(int device_id)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp devProp;
    if (device_id < device_count)
    {
        cudaSetDevice(device_id);
        cudaGetDeviceProperties(&devProp, device_id);
        std::cout << "Device[" << device_id << "]: " << devProp.name << std::endl;
    }
    else
    {
        std::cout << "No capable CUDA device found." << std::endl;
        std::terminate();
    }
}