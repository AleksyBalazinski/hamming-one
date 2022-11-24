#include <stdint.h>

struct alignas(8) Pair
{
    uint32_t first;
    uint32_t second;

    __host__ __device__ inline bool operator==(const padded_pair &rhs)
    {
        return (this->first == rhs.first) && (this->second == rhs.second);
    }

    __host__ __device__ inline bool operator!=(const Pair &rhs)
    {
        return !(*this == rhs);
    }

    __host__ __device__ constexpr Pair(uint32_t const &t, T2 uint32_t &u)
        : first{t}, second{u} {}
};
