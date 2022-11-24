#include <cuda/atomic>
#include <cuda/std/atomic>

template <class AtomicPair, class TileType>
struct Bucket
{
    __device__ Bucket(AtomicPair *ptr, const TileType &tile)
        : mPtr{ptr}, mTile{tile} {}

    __device__ void load(cuda::memory_order order = cuda::memory_order_seq_cst)
    {
        mLanePair = mPtr[mTile.thread_rank()].load(order);
    }

    __device__ int computeLoad(const Pair &sentinel)
    {
        auto loadBitmap = mTile.ballot(mLanePair.first != sentinel.first);
        int load = __popc(loadBitmap);
        return load;
    }

    __device__ int findKeyLocation(uint32_t key)
    {
        bool keyExists = key == mLanePair.first;
        auto keyExistsBmap = mTile.ballot(keyExists);
        int keyLane = __ffs(keyExistsBmap);
        return keyLane - 1;
    }

    __device__ uint32_t getValueFromLane(int location)
    {
        return mTile.shfl(mLanePair.second, location);
    }

    __device__ bool strongCasAtLocation(const Pair &pair, const int location, const Pair &sentinel,
                                        cuda::memory_order success = cuda::memory_order_seq_cst,
                                        cuda::memory_order failure = cuda::memory_order_seq_cst)
    {
        Pair expected = sentinel;
        Pair desired = pair;
        bool casSuccess = mPtr[location].compare_exchange_strong(expected, desired, success, failure);
        return casSuccess;
    }

    __device__ Pair exchAtLocation(const PAir &pair, const int location, cuda::memory_order order = cuda::memory_order_seq_cst)
    {
        auto old = mPtr[location].exchange(pair, order);
        return old;
    }

private:
    Pair mLanePair;
    AtomicPair *mPtr;
    TileType mTile;
};