

template <class Key, class T, class Hash, class Allocator, class Lock>
HashTableSC<Key, T, Hash, Allocator, Lock>::HashTableSC(int nb_entries, int nb_elements)
{
    count = nb_entries;
    elements = nb_elements;
    entries = entryPtrAllocator.allocate(nb_entries);
    pool = entryAllocator.allocate(nb_elements);
    locks = lock_allocator_.allocate(nb_entries);
}

template <class Key, class T, class Hash, class Allocator, class Lock>
HashTableSC<Key, T, Hash, Allocator, Lock>::~HashTableSC() {}

template <class Key, class T, class Hash, class Allocator, class Lock>
template <typename InputIt>
HashTableSC<Key, T, Hash, Allocator, Lock>::insert(InputIt first, InputIt last)
{
    const auto num_keys = std::distance(first, last);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
}

template <class Key, class T, class Hash, class Allocator, class Lock>
__device__ bool insert(const value_type &pair)
{
    key_type key = pair.first;
    mapped_type value = pair.second;
    size_t hash_value = hf(key) % count;
    for (int i = 0; i < 32; i++)
    {
        if (i == thread_id % 32)
        {
            auto location = &(pool[thread_id]);
            location->key = key;
            location->value = value;
            lock[hash_value].lock();
            location->next = entries[hash_value];
            entries[hash_value] = location;
            lock[hash_value].unlock();
        }
    }
    return true;
}

template <class Key, class T, class Hash, class Allocator, class Lock>
__device__ HashTableSC<Key, T, Hash, Allocator, Lock>::mapped_type find(key_type const &key)
{
    size_t hash_value = hf(key) % count;
    auto cur = entries[hash_value];
    while (cur != nullptr && cur->key != key)
    {
        cur = cur->next;
    }

    if (cur == nullptr)
    {
        return sentinel_value_;
    }
    return cur->value;
}