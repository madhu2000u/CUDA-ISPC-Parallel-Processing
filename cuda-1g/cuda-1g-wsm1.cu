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
    int row, col;

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

__global__ void tiledMatrixMultiply(float *a, float *b, float *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedA[BLOCK_DIM * BLOCK_DIM];
    __shared__ float sharedB[BLOCK_DIM * BLOCK_DIM];

    float temp = 0;
    matElement minElement;
    minElement.value = __FLT_MAX__;

    for (int i = 0; i < MATRIX_SIZE / TILE_SZE; i++)
    {
        sharedA[threadIdx.y * TILE_SZE + threadIdx.x] = a[row * MATRIX_SIZE + (i * TILE_SZE + threadIdx.x)];            //index into the global a with the global row (since we are tiling across x dimention of a) and each thread's tile 
        sharedB[threadIdx.y * TILE_SZE + threadIdx.x] = b[(i * TILE_SZE + threadIdx.y) * MATRIX_SIZE + col];            //index into the global b with each thread's tile idexes (since we are tiling across y dimention of b) and globale column 
        __syncthreads();                                                                                                //make sure all values of the sub-matrices are loaded by thre threads before proceding

        for (int j = 0; j < TILE_SZE; j++)
        {
            temp += sharedA[threadIdx.y * TILE_SZE + j] * sharedB[j * TILE_SZE + threadIdx.x];
        }

        __syncthreads();                                                                                                //make sure all sub-matrix calculation is done by threads before advancing to the next sub-matricies
        c[row * MATRIX_SIZE + col] = temp;
        

    }

    
}

int main()
{
    struct timeval start_time, end_time;
    double exec_time;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    CHECK(cudaMallocHost(&d_a, size));
    CHECK(cudaMallocHost(&d_b, size));
    CHECK(cudaMallocHost(&d_c, size));

    matrixInit(h_a, h_b, h_c);

    dim3 blockPerGrid(MATRIX_SIZE / BLOCK_DIM , MATRIX_SIZE / BLOCK_DIM);
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    
    gettimeofday(&start_time, NULL);

    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice));

    tiledMatrixMultiply<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    gettimeofday(&end_time, NULL);

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    std::cout<<"Execution time - "<<exec_time<<std::endl;
    
    std::cout<<"Matrix size - "<<MATRIX_SIZE<<std::endl;

    std::cout<<"Min value - "<<h_c[3503 * MATRIX_SIZE + 2431]<<std::endl;


}