#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <sstream>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " [file1] [file2]\n";
        return -1;
    }
    std::ifstream file1((const char *)argv[1]);
    if (file1.fail())
    {
        std::cerr << "Failed to open " << argv[1] << '\n';
        return -1;
    }
    std::ifstream file2((const char *)argv[2]);
    if (file2.fail())
    {
        std::cerr << "Failed to open " << argv[2] << '\n';
        return -1;
    }

    std::vector<std::pair<int, int>> file1Contents;
    std::vector<std::pair<int, int>> file2Contents;

    int a, b;
    while (file1 >> a >> b)
    {
        file1Contents.push_back(std::make_pair(a, b));
    }

    while (file2 >> a >> b)
    {
        file2Contents.push_back(std::make_pair(a, b));
    }

    if (file1Contents.size() != file2Contents.size())
    {
        std::cout << "Files have different lengths "
                  << "(file1 " << file1Contents.size() << ", "
                  << "file2 " << file2Contents.size() << ")\n";
        std::cout << "TEST FAILED\n";
        return 0;
    }

    std::sort(file1Contents.begin(), file1Contents.end());
    std::sort(file2Contents.begin(), file2Contents.end());

    for (int i = 0; i < file1Contents.size(); i++)
    {
        if (file1Contents[i] != file2Contents[i])
        {
            std::cout << "Difference at line " << i + 1
                      << ". Stop.\n";
            std::cout << "TEST FAILED\n";
            return 0;
        }
    }
    std::cout << "TEST PASSED\n";
}