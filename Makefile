hamming:
	nvcc ./src/kernel.cu -o ./bin/$@ -I ./include

hashTableTest:
	nvcc ./test/hashTableTest.cu -o ./bin/$@ -I ./include
