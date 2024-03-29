/*
 *   Copyright 2023 Aleksy Balazinski (added hashing function for triples)
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

#include "triple.cuh"

template <typename Key>
struct CudaHash {
    using key_type = Key;
    using result_type = Key;
    __host__ __device__ constexpr CudaHash(uint32_t hash_x, uint32_t hash_y)
        : hash_x_(hash_x), hash_y_(hash_y) {}

    constexpr result_type __host__ __device__
    operator()(const key_type& key) const {
        return (((hash_x_ ^ key) + hash_y_) % prime_divisor);
    }

    CudaHash(const CudaHash&) = default;
    CudaHash() = default;
    CudaHash(CudaHash&&) = default;
    CudaHash& operator=(CudaHash const&) = default;
    CudaHash& operator=(CudaHash&&) = default;
    ~CudaHash() = default;

    static constexpr uint32_t prime_divisor = 4294967291u;

private:
    uint32_t hash_x_;
    uint32_t hash_y_;
};

template <typename T1, typename T2, typename T3>
struct CudaHash<triple<T1, T2, T3>> {
    using key_type = triple<T1, T2, T3>;
    using result_type = size_t;
    __host__ __device__ constexpr CudaHash(uint32_t hash_x,
                                           uint32_t hash_y,
                                           uint32_t hash_z,
                                           uint32_t hash_w)
        : hash_x_{hash_x}, hash_y_{hash_y_}, hash_z_{hash_z}, hash_w_{hash_w} {}

    constexpr result_type __host__ __device__
    operator()(const key_type& key) const {
        return ((hash_x_ ^ key.first + hash_y_ ^ key.second + hash_z_ ^
                 key.third + hash_w_) %
                prime_divisor);
    }

    CudaHash(const CudaHash&) = default;
    CudaHash() = default;
    CudaHash(CudaHash&&) = default;
    CudaHash& operator=(CudaHash const&) = default;
    CudaHash& operator=(CudaHash&&) = default;
    ~CudaHash() = default;

    static constexpr uint32_t prime_divisor = 4294967291u;

private:
    uint32_t hash_x_;
    uint32_t hash_y_;
    uint32_t hash_z_;
    uint32_t hash_w_;
};