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

template <typename T1, typename T2, typename T3>
struct triple {
    T1 first;
    T2 second;
    T3 third;

    triple() = default;
    ~triple() = default;
    triple(triple const&) = default;
    triple(triple&&) = default;
    triple& operator=(triple const&) = default;
    triple& operator=(triple&&) = default;

    __host__ __device__ inline bool operator==(const triple& rhs) const {
        return (this->first == rhs.first) && (this->second == rhs.second) &&
               (this->third == rhs.third);
    }
    __host__ __device__ inline bool operator!=(const triple& rhs) const {
        return !(*this == rhs);
    }

    __host__ __device__ constexpr triple(T1 const& t, T2 const& u, T3 const& v)
        : first{t}, second{u}, third{v} {}
};