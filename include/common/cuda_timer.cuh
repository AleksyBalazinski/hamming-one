#pragma once

struct CudaTimer {
    CudaTimer() : start_{}, stop_{}, elapsed_time_{} {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    void startTimer() { cudaEventRecord(start_, 0); }

    void stopTimer() { cudaEventRecord(stop_, 0); }

    float getElapsedMs() {
        getTimeDiffMs();
        return elapsed_time_;
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    void getTimeDiffMs() {
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_time_, start_, stop_);
    }

    cudaEvent_t start_;
    cudaEvent_t stop_;
    float elapsed_time_;
};