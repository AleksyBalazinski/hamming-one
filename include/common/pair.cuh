/*
 *   Copyright 2023 Aleksy Balazinski (removed aliases and aligning)
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

template <class T1, class T2>
struct Pair {
    T1 first;
    T2 second;
    using first_type = T1;
    using second_type = T2;

    Pair() = default;
    ~Pair() = default;
    Pair(Pair const&) = default;
    Pair(Pair&&) = default;
    Pair& operator=(Pair const&) = default;
    Pair& operator=(Pair&&) = default;

    __host__ __device__ inline bool operator==(const Pair& rhs) {
        return (this->first == rhs.first) && (this->second == rhs.second);
    }
    __host__ __device__ inline bool operator!=(const Pair& rhs) {
        return !(*this == rhs);
    }

    __host__ __device__ constexpr Pair(T1 const& t, T2 const& u)
        : first{t}, second{u} {}
};