#include <stdio.h>

int main(){
	int num;
	cudaGetDeviceCount(&num);
	printf("%d gpus available\n", num);
	
	for (int i = 0; i < num; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d has compute capability %d %d\n", prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerMultiProcessor, prop.multiProcessorCount);
	}
	
}

