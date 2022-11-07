#include "Utils.h"

#include <fstream>

void readDataFile(std::string path, Array<bool> *sequences, int numOfSequences)
{
    std::ifstream in;
    in.open(path, std::ios::binary | std::ios::in);

    unsigned char byte = 0x0;
    for (long seq = 0; seq < numOfSequences; seq++)
    {
        for (int j = 0; j < sequences[0].size(); j++)
        {
            in.read((char *)&byte, 1);
            sequences[seq][j] = byte;
        }
    }
    in.close();
}

void readMetadataFile(std::string path, int &numOfSequences, int &sequenceLength)
{
    std::ifstream in;
    in.open(path, std::ios::binary | std::ios::in);
    in >> numOfSequences >> sequenceLength;
}

void printSequences(Array<bool> *sequences, size_t numOfSequences, std::ostream &out)
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