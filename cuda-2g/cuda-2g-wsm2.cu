#include <iostream>
#include <sys/time.h>

#define MATRIX_SIZE 4096
#define BLOCK_DIM 32                
#define TILE_SZE BLOCK_DIM          //Tile size is same as block dimension. defined for better code understandability

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess)\
    {\
        std::cout<<"Error: "<<__FILE__<<":"<<__LINE__<<std::endl;\
        std::cout<<"Code: "<<error<<", reason: "<<cudaGetErrorString(error)<<std::endl;\
        exit(1);\
    }\
}

typedef struct
{
    float value;
    int16_t row, col;

} matElement;

void matrixInit(float *a, float *b, float *c)
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            a[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            b[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            c[i * MATRIX_SIZE + j] = (float)0;
        }
        
    }
}

__device__ void reduceLastWarp(volatile matElement *newSharedB, int threadId)
{
    if(newSharedB[threadId].value > newSharedB[threadId + 32].value){
        newSharedB[threadId].value = newSharedB[threadId + 32].value;
        newSharedB[threadId].row = newSharedB[threadId + 32].row;
        newSharedB[threadId].col = newSharedB[threadId + 32].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 16].value){
        newSharedB[threadId].value = newSharedB[threadId + 16].value;
        newSharedB[threadId].row = newSharedB[threadId + 16].row;
        newSharedB[threadId].col = newSharedB[threadId + 16].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 8].value){
        newSharedB[threadId].value = newSharedB[threadId + 8].value;
        newSharedB[threadId].row = newSharedB[threadId + 8].row;
        newSharedB[threadId].col = newSharedB[threadId + 8].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 4].value){
        newSharedB[threadId].value = newSharedB[threadId + 4].value;
        newSharedB[threadId].row = newSharedB[threadId + 4].row;
        newSharedB[threadId].col = newSharedB[threadId + 4].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 2].value){
        newSharedB[threadId].value = newSharedB[threadId + 2].value;
        newSharedB[threadId].row = newSharedB[threadId + 2].row;
        newSharedB[threadId].col = newSharedB[threadId + 2].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 1].value){
        newSharedB[threadId].value = newSharedB[threadId + 1].value;
        newSharedB[threadId].row = newSharedB[threadId + 1].row;
        newSharedB[threadId].col = newSharedB[threadId + 1].col;
    }
}

__device__ void minBlockReduce(matElement *newSharedB, int threadId)
{
    for (unsigned int stride = (BLOCK_DIM * BLOCK_DIM)/2; stride > 32; stride >>= 1)
    {
        if(threadId < stride)
        {
            if(newSharedB[threadId].value > newSharedB[threadId + stride].value){
                newSharedB[threadId] = newSharedB[threadId + stride];
            }
        }
        __syncthreads();
    }
    if(threadId < 32) reduceLastWarp(newSharedB, threadId);
}

__global__ void find2Min(int16_t firstMinRow, int16_t firstMinCol, float *c, matElement *d_minValueFromEachBlock)
{
    int16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int16_t col = blockIdx.x * blockDim.x + threadIdx.x;

    int16_t threadId = threadIdx.y * BLOCK_DIM + threadIdx.x;

    __shared__ matElement sharedC[BLOCK_DIM * BLOCK_DIM];

    if(row == 0 && col == 0) c[firstMinRow * MATRIX_SIZE + firstMinCol] = __FLT_MAX__;
    __syncthreads();

    sharedC[threadId].value = c[row * MATRIX_SIZE + col];
    sharedC[threadId].row = row;
    sharedC[threadId].col = col;
    __syncthreads();

    minBlockReduce(sharedC, threadId);
    if(threadId == 0){   
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].value = sharedC[0].value;
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].row = sharedC[0].row;
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].col = sharedC[0].col;
    }
}

__global__ void tiledMatrixMultiply(int gpu, float *a, float *b, float *c, matElement *d_minValueFromEachBlock)
{
    int16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int16_t col = blockIdx.x * blockDim.x + threadIdx.x;

    int16_t threadId = threadIdx.y * BLOCK_DIM + threadIdx.x;

    __shared__ float sharedA[BLOCK_DIM * BLOCK_DIM];
    __shared__ float sharedB[BLOCK_DIM * BLOCK_DIM * sizeof(matElement)];                    

    float temp = 0;

    for (int i = 0; i < 128; i++)
    {
        sharedA[threadId] = a[row * MATRIX_SIZE + (i * TILE_SZE + threadIdx.x)];                 //index into the global a with the global row (since we are tiling across x dimention of a) and each thread's tile 
        sharedB[threadId] = b[(i * TILE_SZE + threadIdx.y) * MATRIX_SIZE + col];                 //index into the global b with each thread's tile idexes (since we are tiling across y dimention of b) and globale column 
        __syncthreads();                                                                         //make sure all values of the sub-matrices are loaded by thre threads before proceding

        for (int j = 0; j < TILE_SZE; j++)
        {
            temp += sharedA[threadIdx.y * TILE_SZE + j] * sharedB[j * TILE_SZE + threadIdx.x];
        }

        __syncthreads();                                                                         //make sure all sub-matrix calculation is done by threads before advancing to the next sub-matricies

    }
    matElement *newSharedB = (matElement*) sharedB;                                              //reuse shared mem for finding min element

    newSharedB[threadId].value = temp;
    newSharedB[threadId].row = row;
    newSharedB[threadId].col = col;
    __syncthreads();
    
    c[row * 4096 + col] = temp;

    minBlockReduce(newSharedB, threadId);
    if(threadId == 0){   
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].value = newSharedB[0].value;
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].row = newSharedB[0].row;
        d_minValueFromEachBlock[blockIdx.y * 128 + blockIdx.x].col = newSharedB[0].col;
    }

}

int main()
{   
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    int rowSizePerGpu = ((MATRIX_SIZE - 1) / num_gpus) + 1;

    struct timeval start_time, end_time;
    double exec_time;
    matElement minElement[num_gpus * 2];

    for (int i = 0; i < num_gpus * 2; i++)
    {
        minElement[i].value = __FLT_MAX__;
    }

    float *h_a, *h_b, *h_c;
    float *d_a[num_gpus], *d_b, *d_c[num_gpus];             //d_b & h_b same for both gpus

    cudaStream_t streams[num_gpus];

    matElement *h_minValueFromEachBlock[num_gpus];
    matElement *d_minValueFromEachBlock[num_gpus];

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    int gridSize = rowSizePerGpu / BLOCK_DIM;

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    

    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);

        h_minValueFromEachBlock[i] = (matElement*)malloc((128) * (gridSize) * sizeof(matElement));      //128 because there are 128 block across x axis and 128 blocks * 32 block size = 4096 columns across x. similarly gridSize (here 64) blocks along y axis * 32 block size = 2048 rows per gpu

        CHECK(cudaMallocHost(&d_a[i], rowSizePerGpu * MATRIX_SIZE * sizeof(float)));                    //each gpu gets rowSizePerGpu * MATRIX_SIZE (mxn) matrix for a
        CHECK(cudaMallocHost(&d_b, size));                                                              //entire b matrix (nxn) is required for matrix multipliaion -_-
        CHECK(cudaMallocHost(&d_c[i], rowSizePerGpu * MATRIX_SIZE * sizeof(float)));                    //resulting matrix per gpu is (mxn)
        CHECK(cudaMallocHost(&d_minValueFromEachBlock[i], (128) * (gridSize) * sizeof(matElement)));

        cudaStreamCreate(&streams[i]);
    }

    matrixInit(h_a, h_b, h_c);
    
    dim3 blockPerGrid(128 , gridSize);
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    
    gettimeofday(&start_time, NULL);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        CHECK(cudaMemcpyAsync(d_a[i], h_a + i * rowSizePerGpu * MATRIX_SIZE, rowSizePerGpu * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, streams[i]));
        
        tiledMatrixMultiply<<<blockPerGrid, threadsPerBlock, 0, streams[i]>>>(i, d_a[i], d_b, d_c[i], d_minValueFromEachBlock[i]);

        CHECK(cudaMemcpyAsync(h_c + i * rowSizePerGpu * MATRIX_SIZE, d_c[i], rowSizePerGpu * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        CHECK(cudaMemcpyAsync(h_minValueFromEachBlock[i], d_minValueFromEachBlock[i], (128) * (gridSize) * sizeof(matElement), cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    for (int j = 0; j < num_gpus; j++)
    {
        for (int i = 0; i < (128) * (gridSize); i++)
        {
            if(h_minValueFromEachBlock[j][i].value < minElement[j * num_gpus].value)
            {   
                minElement[j * num_gpus].value = h_minValueFromEachBlock[j][i].value;           
                minElement[j * num_gpus].row = h_minValueFromEachBlock[j][i].row + (j * rowSizePerGpu);         //each gpu uses it's own row, col addressing from 0. but gpu 1 maybe hadling rows from 2048 which it considers as rows from 0 and finds its min value index so we scale it for each gpu here
                minElement[j * num_gpus].col = h_minValueFromEachBlock[j][i].col;
            }
        }
    }

    for(int j = 0; j < num_gpus; j++)
    {   
        cudaSetDevice(j);
        find2Min<<<blockPerGrid, threadsPerBlock, 0, streams[j]>>>((minElement[j * num_gpus].row - (j * rowSizePerGpu)), minElement[j * num_gpus].col, d_c[j], d_minValueFromEachBlock[j]);

        CHECK(cudaMemcpyAsync(h_minValueFromEachBlock[j], d_minValueFromEachBlock[j], (128) * (gridSize) * sizeof(matElement), cudaMemcpyDeviceToHost, streams[j]));
    }

    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int j = 0; j < num_gpus; j++)
    {    
        for (int i = 0; i < (128) * (gridSize); i++)
        {
            if(h_minValueFromEachBlock[j][i].value < minElement[j * num_gpus + 1].value)
            {
                minElement[j * num_gpus + 1].value = h_minValueFromEachBlock[j][i].value;           
                minElement[j * num_gpus + 1].row = h_minValueFromEachBlock[j][i].row + (j * rowSizePerGpu);
                minElement[j * num_gpus + 1].col = h_minValueFromEachBlock[j][i].col;
            }
        }
    }
    
    matElement finalMinElements[2];
    float minVal = __FLT_MAX__;
    int minIndex = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < num_gpus * 2; j++)
        {
            if(minElement[j].value < minVal)
            {
                minVal = minElement[j].value;
                minIndex = j;
            }
        }
        finalMinElements[i] = minElement[minIndex];
        minVal = __FLT_MAX__;
        minElement[minIndex].value = __FLT_MAX__;
    }
    
    gettimeofday(&end_time, NULL);

    //Free device memory
    for (int i = 0; i < num_gpus; i++)
    {
        cudaFree(d_a[i]);
        cudaFree(d_b);
        cudaFree(d_c[i]);
        cudaFree(d_minValueFromEachBlock[i]);
    } 

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    std::cout<<"Execution time - "<<exec_time<<std::endl;
    
    std::cout<<"Matrix size - "<<MATRIX_SIZE<<std::endl;

    std::cout<<"Min value 1 (val, row, col) - ("<<finalMinElements[0].value<<", "<<finalMinElements[0].row<<", "<<finalMinElements[0].col<<")"<<std::endl;

    std::cout<<"Min value 2 (val, row, col) - ("<<finalMinElements[1].value<<", "<<finalMinElements[1].row<<", "<<finalMinElements[1].col<<")"<<std::endl;


}