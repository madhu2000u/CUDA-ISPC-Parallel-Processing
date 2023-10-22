#include <iostream>
#include <sys/time.h>

#define MATRIX_SIZE 4096

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

__global__ void matrixMultiply(float *a, float *b, float *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0;
    matElement minElement;
    minElement.value = __FLT_MAX__;
    
    if(row < MATRIX_SIZE && col < MATRIX_SIZE)
    {
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            temp += a[row * MATRIX_SIZE + i] * b[i * MATRIX_SIZE + col];
        }
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

    cudaMallocHost(&d_a, size);
    cudaMallocHost(&d_b, size);
    cudaMallocHost(&d_c, size);

    matrixInit(h_a, h_b, h_c);

    dim3 blockPerGrid(MATRIX_SIZE / 32 , MATRIX_SIZE / 32);
    dim3 threadsPerBlock(32, 32);

    
    gettimeofday(&start_time, NULL);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

    matrixMultiply<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    gettimeofday(&end_time, NULL);

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    std::cout<<"Execution time - "<<exec_time<<std::endl;
    
    std::cout<<"Matrix size - "<<MATRIX_SIZE<<std::endl;

    std::cout<<"Min value - "<<h_c[3503 * MATRIX_SIZE + 2431]<<std::endl;


}