#include "Pair.cuh"
#include <cuda/atomic>
#include "detail/kernels.cuh"
#include <memory>

template <class Hash>
struct HashMap
{
    using AtomicPair = cuda::atomic<Pair, cuda::thread_scope_device>;
    static constexpr int bucketSize = 16;

    HashMap(size_t capacity, uint32_t sentinelKey, uint32_t sentinelValue);

    template <class InputIt>
    bool insert(InputIt first, InputIt last);

    template <class InputIt, class OutputIt>
    void find(InputIt first, InputIt last, OutputIt outBegin);

    template <typename TileType>
    __device__ bool insert(const Pair &pair, const TileType &tile);

    template <class TileType>
    __device__ uint32_t find(uint32_t key, const TileType &tile);

    template <typename RNG>
    void randomizeHashFunctions(RNG &rng);

    // size_t size(); // TODO

private:
    template <typename InputIt, typename HashMap>
    friend __global__ void kernels::tiledInsertKernel(InputIt, InputIt, HashMap);

    template <typename InputIt, typename OutputIt, typename HashMap>
    friend __global__ void kernels::tiledFindKernel(InputIt,
                                                    InputIt,
                                                    OutputIt,
                                                    HashMap);

    // template <int BlockSize, typename InputT, typename HashMap>
    // friend __global__ void kernels::count_kernel(const InputT,
    //                                              std::size_t *,
    //                                              HashMap); // TODO

    size_t mCapacity;
    uint32_t mSentinelKey{};
    uint32_t mSentinelValue{};

    AtomicPair *mDevTable{};
    std::shared_ptr<AtomicPair> mTable;

    bool *mDevBuildSuccess;
    std::shared_ptr<bool> mBuildSuccess;

    uint32_t mMaxCuckooChains;

    Hash mHf0;
    Hash mHf1;
    Hash mHf2;

    size_t mNumBuckets;
};
