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

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

template <typename RandStateIt>
__global__ void init(RandStateIt rs_first, RandStateIt rs_last) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    auto count = rs_last - rs_first;
    if (thread_id >= count)
        return;
    curand_init(1337, thread_id, 0, &rs_first[thread_id]);
}

template <typename RandStateIt, typename OutputIt>
__global__ void generateSequences(RandStateIt rs_first,
                                  RandStateIt rs_last,
                                  OutputIt out_first) {
    auto total_len = rs_last - rs_first;
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= total_len)
        return;

    float ret = curand_uniform(&rs_first[thread_id]);
    if (ret < 0.5)
        out_first[thread_id] = 0;
    else
        out_first[thread_id] = 1;
}

void generateMetadataFile(std::string path, int l, int n) {
    std::ofstream out;
    out.open(path, std::ios::out);
    out << n << ' ' << l;
}

template <typename InputIt>
void generateDataFile(std::string path,
                      int l,
                      int n,
                      InputIt sequences_first,
                      InputIt sequences_last) {
    std::ofstream out;
    out.open(path, std::ios::out);
    for (long seq = 0; seq < n; seq++) {
        for (int j = 0; j < l; j++) {
            out << sequences_first[seq * l + j] << ' ';
        }
        out << '\n';
    }
    out.close();
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " [path to metadata file] [path to data file]"
                  << "[sequence length] [# of sequences]\n";
        return -1;
    }
    std::string path_to_metadata(argv[1]);
    std::string path_to_data(argv[2]);

    int seq_length = std::stoi(argv[3]);
    long num_sequences = std::stol(argv[4]);

    long total_length = seq_length * num_sequences;
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (total_length + block_size - 1) / block_size;

    thrust::device_vector<int> d_sequences(total_length);
    // thrust::device_vector<curandState> d_states(total_length);
    curandState* d_states;
    cudaMalloc(&d_states, total_length * sizeof(curandState));

    init<<<num_blocks, block_size>>>(d_states, d_states + total_length);
    generateSequences<<<num_blocks, block_size>>>(
        d_states, d_states + total_length, d_sequences.begin());
    thrust::device_vector<int> h_sequences = d_sequences;

    generateDataFile(path_to_data, seq_length, num_sequences,
                     d_sequences.begin(), d_sequences.end());
    generateMetadataFile(path_to_metadata, seq_length, num_sequences);

    cudaFree(d_states);
}