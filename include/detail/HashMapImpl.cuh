#include "HashMap.hpp"
#include "bucket.cuh"
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <random>
#include "rng.h"

template <class T>
struct CudaDeleter
{
    void operator()(T *p) { cudaFree(p); }
};

template <class Hash>
HashMap<Hash>::HashMap(size_t capacity, uint32_t emptyKeySentinel, uint32_t emptyValueSentinel)
    : mCapacity{std::max(capacity, size_t{1})},
      mSentinelKey{emptyKeySentinel},
      mSentinelValue{emptyValueSentinel}
{
    auto reminder = mCapacity % bucketSize;
    if (reminder)
    {
        mCapacity += bucketSize - reminder;
    }

    mNumBuckets = mCapacity / bucketSize;
    cudaMalloc(&mDevTable, mCapacity);
    mTable = std::shared_ptr<AtomicPair>(mDevTable, CudaDeleter<AtomicPair>());

    cudaMalloc(&mDevBuildSuccess, 1);
    mBuildSuccess = std::shared_ptr<bool>(mDevBuildSuccess, CudaDeleter<bool>());

    Pair emptyPair{mSentinelKey, mSentinelValue};
    thrust::fill(thrust::device, mDevTable, mDevTable + mCapacity, emptyPair);

    double lgInputSize = (float)(log((double)capacity) / log(2.0));
    const unsigned maxIter = 7;
    mMaxCuckooChains = static_cast<uint32_t>(maxIter * lgInputSize);

    std::mt19937 rng(2);
    mHf0 = initializeHf<Hash>(rng); // TODO
    mHf1 = initializeHf<Hash>(rng);
    mHf2 = initializeHf<Hash>(rng);

    bool success = true;
    cudaMemcpy(mDevBuildSuccess, &success, sizeof(bool), cudaMemcpyHostToDevice);
}

template <class Hash>
template <class InputIt>
bool HashMap<Hash>::insert(InputIt first, InputIt last)
{
    const auto numKeys = std::distance(first, last);

    const uint32_t blockSize = 128;
    const uint32_t numBlocks = (numKeys + blockSize - 1) / blockSize;
    kernels::tiledInsertKernel<<<numBlocks, blockSize>>>(first, last, *this);
    bool success;
    cudaMemcpyAsync(&success, mDevBuildSuccess, sizeof(bool), cudaMemcpyDeviceToHost, 0);

    return success;
}

template <class Hash>
template <class InputIt, class OutputIt>
void HashMap<Hash>::find(InputIt first, InputIt last, OutputIt outBegin)
{
    const auto numKeys = last - first;
    const uint32_t blockSize = 128;
    const uint32_t numBlocks = (numKeys + blockSize - 1) / blockSize;

    kernels::tiledFindKernel<<<numBlocks, blockSize>>>(
        first, last, output_begin, *this);
}

template <class Hash>
template <class TileType>
__device__ bool HashMap<Hash>::insert(const Pair &pair, const TileType &tile)
{
    mars_rng_32 rng;

    auto bucketId = mHf0(pair.first) % mNumBuckets;

    uint32_t cuckooCounter = 0;
    auto laneId = tile.thread_rank();
    const int electedLane = 0;
    Pair sentinelPair{mSentinelKey, mSentinelValue};
    Pair insertionPair = pair;

    using BucketType = Bucket<AtomicPair, TileType>;
    do
    {
        BucketType curBucket(&mDevTable[bucketId * bucketSize], tile);
        curBucket.load(cuda::memory_order_relaxed);
        int load = curBucket.computeLoad(sentinelPair);
        if (load != bucketSize)
        {
            bool casSuccess = false;
            if (laneId == electedLane)
            {
                casSuccess = curBucket.strongCasAtLocation(insertionPair, load, sentinelPair, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
            }
            casSuccess = tile.shfl(casSuccess, electedLane);
            if (casSuccess)
                return true;
        }
        else
        {
            if (laneId = electedLane)
            {
                auto randomLocation = rng() % bucketSize;
                auto oldPair = curBucket.exchAtLocation(insertionPair, randomLocation, cuda::memory_order_relaxed);

                auto bucket0 = mHf0(oldPair.first) % mNumBuckets;
                auto bucket1 = mHf1(oldPair.first) & mNumBuckets;
                auto bucket2 = mHf2(oldPair.first) & mNumBuckets;

                auto newBucketId = bucket0;
                newBucketId = bucketId == bucket1 ? bucket2 : newBucketId;
                newBucketId = bucketId == bucket0 ? bucket1 : newBucketId;

                bucketId = newBucketId;
                insertionPair = oldPair;
            }
            bucketId = tile.shfl(bucketId, electedLane);
            cuckooCounter++;
        }
    } while (cuckooCounter < mMaxCuckooChains);
    return false;
}

template <class Hash>
template <class TileType>
__device__ uint32_t HashMap<Hash>::find(uint32_t key, const TileType &type)
{
    const int numHfs = 3;
    auto bucketId = mHf0(key) % mNumBuckets;
    Pair sentinelPair{mSentinelKey, mSentinelValue};
    using BucketType = Bucket<AtomicPair, TileType>;
    for (int hf = 0; hf < numHfs; hf++)
    {
        BucketType curBucket(&mDevTable[bucketId * bucketSize], tile);
        curBucket.load(cuda::memory_order_relaxed);
        int keyLocation = curBucket.findKeyLocation(key);
        if (keyLocation != -1)
        {
            auto foundValue = curBucket.getValueFromLane(keyLocation);
            return foundValue;
        }
        else if (curBucket.computeLoad(sentinelPair) < bucketSize)
        {
            return mSentinelValue;
        }
        else
        {
            bucketId = hf == 0 ? mHf1(key) % mNumBuckets : mHf2(key) % mNumBuckets;
        }

        return mSentinelValue;
    }
}

template <class Hash>
template <class RNG>
void HashMap<Hash>::randomizeHashFunctions(RNG &r)
{
    mHf0 = intialize_hf<Hash>(rng); // TODO
    mHf1 = initialize_hf<Hash>(rng);
    mHf2 = initialize_hf<Hash>(rng);
}