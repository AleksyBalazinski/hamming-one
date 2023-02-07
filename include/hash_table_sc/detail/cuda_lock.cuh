/**
 *   Copyright 2023 Aleksy Balazinski
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

#include <cuda/atomic>
#include <cuda/std/atomic>

class CudaLock {
    cuda::std::atomic_flag mutex_ = ATOMIC_FLAG_INIT;

public:
    __device__ void lock() {
        while (mutex_.test_and_set(cuda::memory_order_acquire))
            ;
    }
    __device__ void unlock() { mutex_.clear(cuda::memory_order_release); }
};