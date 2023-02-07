/*
 *   Copyright 2023 Aleksy Balazinski
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

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