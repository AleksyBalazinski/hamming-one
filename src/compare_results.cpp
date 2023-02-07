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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [file1] [file2]\n";
        return -1;
    }
    std::ifstream file1((const char*)argv[1]);
    if (file1.fail()) {
        std::cerr << "Failed to open " << argv[1] << '\n';
        return -1;
    }
    std::ifstream file2((const char*)argv[2]);
    if (file2.fail()) {
        std::cerr << "Failed to open " << argv[2] << '\n';
        return -1;
    }

    std::vector<std::pair<int, int>> file1Contents;
    std::vector<std::pair<int, int>> file2Contents;

    int a, b;
    while (file1 >> a >> b) {
        file1Contents.push_back(std::make_pair(a, b));
    }

    while (file2 >> a >> b) {
        file2Contents.push_back(std::make_pair(a, b));
    }

    if (file1Contents.size() != file2Contents.size()) {
        std::cout << "Files have different lengths "
                  << "(file1 " << file1Contents.size() << ", "
                  << "file2 " << file2Contents.size() << ")\n";
        std::cout << "TEST FAILED\n";
        return 0;
    }

    std::sort(file1Contents.begin(), file1Contents.end());
    std::sort(file2Contents.begin(), file2Contents.end());

    for (int i = 0; i < file1Contents.size(); i++) {
        if (file1Contents[i] != file2Contents[i]) {
            std::cout << "Difference at line " << i + 1 << ". Stop.\n";
            std::cout << "TEST FAILED\n";
            return 0;
        }
    }
    std::cout << "TEST PASSED\n";
}