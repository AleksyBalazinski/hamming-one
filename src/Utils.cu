#include "Utils.h"

#include <fstream>

void readDataFile(std::string path, Array<int> *sequences, int numOfSequences)
{
    std::ifstream in;
    in.open(path, std::ios::in);

    for (long seq = 0; seq < numOfSequences; seq++)
    {
        for (int j = 0; j < sequences[0].size(); j++)
        {
            in >> sequences[seq][j];
        }
    }
    in.close();
}

void readDataFromFile(std::string path, int *sequences, int numOfSequences, int seqLength)
{
    std::ifstream in;
    in.open(path, std::ios::in);

    for (size_t i = 0; i < numOfSequences * seqLength; i++)
    {
        in >> sequences[i];
    }
    in.close();
}

void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength)
{
    std::ifstream in;
    in.open(path, std::ios::in);
    in >> numOfSequences >> sequenceLength;
}

void printSequences(Array<int> *sequences, size_t numOfSequences, std::ostream &out)
{
    for (long seq = 0; seq < numOfSequences; seq++)
    {
        out << seq << ": ";
        for (int j = 0; j < sequences[0].size(); j++)
        {
            out << sequences[seq][j];
        }
        out << '\n';
    }
}

void printSequences(int *sequences, int numOfSequences, int seqLength)
{
    for (int seq = 0; seq < numOfSequences; seq++)
    {
        printf("%d: ", seq);
        for (int j = 0; j < seqLength; j++)
        {
            printf("%d", sequences[seq * seqLength + j]);
        }
        printf("\n");
    }
}
