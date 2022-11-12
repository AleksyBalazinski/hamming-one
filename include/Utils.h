#pragma once

#include <string>
#include "Array.h"

// void readDataFile(std::string path, Array<int> *sequences, int numOfSequences);
void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength);
// void printSequences(Array<int> *sequences, size_t numOfSequences, std::ostream &out);
void readDataFromFile(std::string path, int *sequences, int numOfSequences, int seqLength);
void printSequences(int *sequences, int numOfSequences, int seqLength);
