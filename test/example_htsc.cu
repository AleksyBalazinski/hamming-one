#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <limits>
#include <vector>
#include "common/pair.cuh"
#include "hash_table_sc/hash_table_sc.hpp"

int main() {
    using key_type = uint32_t;
    using value_type = uint32_t;
    using pair_type = Pair<key_type, value_type>;
    size_t num_keys = 64;
    key_type invalid_key = std::numeric_limits<key_type>::max();
    value_type invalid_value = std::numeric_limits<value_type>::max();

    HashTableSC<key_type, value_type> table(128, num_keys, invalid_value);

    std::vector<key_type> h_keys;
    std::vector<value_type> h_values;
    std::vector<pair_type> h_pairs;
    for (int i = 0; i < num_keys; i++) {
        h_keys.push_back(i);
        h_values.push_back(2 * i + 1);
        h_pairs.emplace_back(h_keys[i], h_values[i]);
    }

    thrust::device_vector<pair_type> d_pairs(num_keys);
    d_pairs = h_pairs;

    auto input_start = d_pairs.data().get();
    auto input_last = input_start + num_keys;
    table.insert(input_start, input_last);

    thrust::device_vector<key_type> d_queries(num_keys);
    std::vector<key_type> h_queries(h_keys);
    h_queries[5] = 1234;  // not in the hash table

    d_queries = h_queries;
    thrust::device_vector<value_type> d_results(num_keys);
    auto queries_start = d_queries.data().get();
    auto queries_last = queries_start + num_keys;
    auto output_start = d_results.data().get();
    table.find(queries_start, queries_last, output_start);

    thrust::host_vector<value_type> h_results = d_results;
    for (int i = 0; i < num_keys; i++) {
        auto key = h_keys[i];
        auto expected_pair = h_pairs[i];
        auto found_result = h_results[i];
        std::cout << "Found " << key << " -> " << found_result << '\n';
    }
}
