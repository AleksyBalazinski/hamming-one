#pragma once

#include <fstream>
#include <string>
#include <vector>

template <typename T>
void readDataFile(std::string path, std::vector<std::vector<T>>& sequences) {
    int numOfSequences = sequences.size();
    int seqLength = sequences[0].size();
    std::ifstream in;
    in.open(path, std::ios::in);

    for (int seq = 0; seq < numOfSequences; seq++) {
        for (int j = 0; j < seqLength; j++) {
            in >> sequences[seq][j];
        }
    }
    in.close();
}

template <typename T>
void readRefinedDataFile(std::string path,
                  std::vector<T>& sequences,
                  int seq_length,
                  std::vector<T>& seq_ids) {
    std::ifstream in;
    in.open(path, std::ios::in);

    int num_sequences = sequences.size() / seq_length;
    for (int i = 0; i < num_sequences; i++) {
        int seq_id;
        in >> seq_id;
        for(int j=0; j < seq_length; j++) {
            seq_ids[j + i * seq_length] = seq_id;
        }
        for (int j = 0; j < seq_length; j++) {
            in >> sequences[j + i * seq_length];
        }
    }
}

template<typename T>
void readRefinedDataFile(std::string path, std::vector<std::vector<T>>& sequences, std::vector<T>& seq_ids) {
    int num_sequences = sequences.size();
    int seq_length = sequences[0].size();

    std::ifstream in;
    in.open(path, std::ios::in);

    for(int i=0;i<num_sequences;i++) {
        in >> seq_ids[i];
        for(int j = 0;j<seq_length;j++) {
            in >> sequences[i][j];
        }
    }
}

template <typename T>
void readDataFile(std::string path, std::vector<T>& sequences) {
    std::ifstream in;
    in.open(path, std::ios::in);

    for (size_t i = 0; i < sequences.size(); i++) {
        in >> sequences[i];
    }
    in.close();
}

template<typename SeqIt, typename SeqIdIt>
void readDataFile(std::string path, int seq_length, SeqIt sequences_first, SeqIt sequences_last, SeqIdIt seq_ids_first) {
    std::ifstream in;
    in.open(path, std::ios::in);

    auto total_len = sequences_last - sequences_first;
    for (size_t i = 0; i < total_len; i++) {
        in >> sequences_first[i];
        seq_ids_first[i] = i / seq_length;
    }
    in.close();
}


void readMetadataFile(std::string path,
                      int& numOfSequences,
                      int& sequenceLength) {
    std::ifstream in;
    in.open(path, std::ios::in);
    in >> numOfSequences >> sequenceLength;
}

void printSequences(int* sequences, int numOfSequences, int seqLength) {
    for (int seq = 0; seq < numOfSequences; seq++) {
        printf("%d: ", seq);
        for (int j = 0; j < seqLength; j++) {
            printf("%d", sequences[seq * seqLength + j]);
        }
        printf("\n");
    }
}