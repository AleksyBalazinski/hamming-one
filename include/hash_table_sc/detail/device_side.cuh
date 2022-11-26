namespace detail::device_side
{
    template <typename InputIt, typename HashTable>
    __global__ void insert(InputIt first, InputIt last, HashTable table)
    {
        int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        auto count = last - first;
        if (thread_id > count - 1)
            return;

        typename HashTable::value_type insertion_pair = first[thread_id];
        table.insert(insertion_pair);
    }

    template <typename InputIt, typename OutputIt, typename HashTable>
    __global__ void find(InputIt first, InputIt last, Output output_begin, HashTable table)
    {
        int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        auto count = last - first;
        if (thread_id > count - 1)
            return;

        typename HashTable::key_type find_key = first[thread_id];
        typename HashTable::mapped_type result = table.find(find_key);

        return result;
    }
}