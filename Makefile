ifeq ($(OS),Windows_NT)
	CCBIN="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx86\x64\cl.exe"
else
	CCBIN=g++
endif

all: hamming hammingcpu bruteforcecpu generator compareResults

hamming:
	nvcc -ccbin $(CCBIN) -O3 ./src/kernel.cu ./src/Utils.cu -o ./bin/$@ -I ./include

hamming2:
	nvcc -ccbin $(CCBIN) -O3 ./src/kernel2.cu ./src/Utils.cu -o ./bin/$@ -I ./include

hamming3:
	nvcc -ccbin $(CCBIN) -O3 ./src/kernel3.cu ./src/Utils.cu -o ./bin/$@ -I ./include

hashTableTest:
	nvcc -ccbin $(CCBIN) ./test/hashTableTest.cu -o ./bin/$@ -I ./include

arrayTest:
	nvcc -ccbin $(CCBIN) ./test/arrayTest.cu -o ./bin/$@ -I ./include

hammingcpu:
	nvcc -x c++ -ccbin $(CCBIN) ./src/linear.cpp -o ./bin/$@ -I ./include

bruteforcecpu:
	g++ ./src/bruteForce.cpp -o ./bin/$@

generator:
	g++ ./src/generator.cpp -o ./bin/$@

compareResults:
	g++ ./test/compareResults.cpp -o ./bin/$@

test_bcht:
	nvcc -gencode arch=compute_75,code=sm_75 --std c++17 --extended-lambda --expt-relaxed-constexpr -ccbin $(CCBIN) ./test/test_bcht.cu -o ./bin/$@ -I ./include
