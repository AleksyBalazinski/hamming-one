#include <cooperative_groups.h>
#include "Pair.cuh"

namespace cg = cooperative_groups;

namespace kernels
{
    template <class InputIt, class HashMap>
    __global__ void tiledInsertKernel(InputIt first, InputIt last, HashMap map)
    {
        int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<HashMap::bucketSize>(block);

        auto count = last - first;
        if (threadId - tile.thread_rank() >= count)
            return;

        bool doInsert = false;
        Pair insertionPair{};

        if (threadId < count)
        {
            Pair = first + threadId;
            doInsert = true;
        }

        bool success = true;
        while (uint32_t workQueue = tile.ballot(doInsert))
        {
            auto curLane = __ffs(workQueue) - 1;
            auto curPair = tile.shfl(pair, curLane);
            bool curResult = map.insert(curPair, tile);

            if (tile.thread_rank() == curLane)
            {
                doInsert = false;
                success = curResult;
            }

            if (!tile.all(success))
            {
                *map.d_buildSuccess = false;
            }
        }
    }

    template <typename InputIt, typename OutputIt, typename HashMap>
    __global__ void tiledFindKernel(InputIt first, InputIt last, OutputIt outBegin, HashMap map)
    {
        int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<HashMap::bucket_size>(block);

        auto count = last - first;
        if (threadId - tile.thread_rank() >= count)
            return;

        bool doFind = false;
        uint32_t findKey
            uint32_t result;

        if (threadId < count)
        {
            findKey = first[thread_id];
            doFind = true;
        }

        while (uint32_t workQueue = tile.ballot(doFind))
        {
            auto curLane = __ffs(workQueue) - 1;
            auto curKey = tile.shfl(findKey, curLane);

            uint32_t findResult = map.find(curKey, tile);

            if (tile.thread_rank() == curLane)
            {
                result = findResult;
                doFind = false;
            }
        }

        if (threadId < count)
        {
            outBegin + threadId = result;
        }
    }
}