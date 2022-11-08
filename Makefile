ifeq ($(OS),Windows_NT)
	CCBIN="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx86\x64\cl.exe"
else
	CCBIN=g++
endif

hamming:
	nvcc -ccbin $(CCBIN) ./src/kernel.cu -o ./bin/$@ -I ./include

hashTableTest:
	nvcc -ccbin $(CCBIN) ./test/hashTableTest.cu -o ./bin/$@ -I ./include

arrayTest:
	nvcc -ccbin $(CCBIN) ./test/arrayTest.cu -o ./bin/$@ -I ./include