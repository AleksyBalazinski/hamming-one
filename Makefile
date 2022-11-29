ifeq ($(OS),Windows_NT)
	CCBIN="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx86\x64\cl.exe"
else
	CCBIN=g++
endif

CFLAGS=-Xcompiler=/std:c++17 -gencode arch=compute_75,code=sm_75 --std c++17 --expt-extended-lambda -extended-lambda --expt-relaxed-constexpr --forward-unknown-to-host-linker --forward-unknown-to-host-compiler
BOOSTDIR="C:\Program Files\boost_1_80_0"

hamming:
	nvcc $(CFLAGS) -ccbin $(CCBIN) ./src/kernel.cu -o ./bin/$@ -I ./include

hammingcpu:
	g++ ./src/linear.cpp -o ./bin/$@ -I ./include

bruteforcecpu:
	g++ ./src/brute_force.cpp -o ./bin/$@ -I ./include

generator:
	g++ ./src/generator.cpp -o ./bin/$@

compare_results:
	g++ ./src/compare_results.cpp -o ./bin/$@

remove_duplicates:
	g++ ./src/remove_duplicates.cpp -o ./bin/$@ -I ./include -I $(BOOSTDIR)

example_htsc:
	nvcc $(CFLAGS) -ccbin $(CCBIN) ./test/example_htsc.cu -o ./bin/$@ -I ./include

extend_to_duplicates:
	g++ ./src/$@.cpp -o ./bin/$@