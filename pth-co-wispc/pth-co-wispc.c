#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <pthread.h>
#include "pth-co-wispc.h"
#include "my_ispc-common.h"

// #define num_threads 32

pthread_mutex_t minCElement_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;

// struct matElement
// {
//     float value;
//     int row, col;
// };

struct matElement minCElementGlobal;
struct matElement minCElementThread[MATRIX_SIZE];
struct thread_args
{   
    float (*A)[MATRIX_SIZE];
    float (*B)[MATRIX_SIZE];
    float (*C)[MATRIX_SIZE];
    int row_start, row_end;
    
};

void printMatrix(float mat[MATRIX_SIZE][MATRIX_SIZE]){
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
           printf("%f    ", mat[i][j]);
            
        }
        printf("\n");
        
    }
}

unsigned long long rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

void checkMatrixResult(float A[MATRIX_SIZE][MATRIX_SIZE], float B[MATRIX_SIZE][MATRIX_SIZE], float result_matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    bool check = true;
    double threshold = 1.0E-1;
    struct matElement ref_min;
    ref_min.value = FLT_MAX;
    float (*ref_matrix)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    // Reference Matrix multiplication
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   ref_matrix[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                //val += A[i][k] * B[j][k];
                ref_matrix[i][j] += A[i][k] * B[j][k];
            }
            if(ref_matrix[i][j] < ref_min.value)
            {
                ref_min.value = ref_matrix[i][j];
                ref_min.row = i;
                ref_min.col = j;
            }
            
        }
        
    }
    #if DEBUG
        printf("\n\nPrinting ref_matrix...\n");
        printMatrix(ref_matrix);
    #endif

    for (int i = 0; i < MATRIX_SIZE && check; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            //printf("\t%f\t", fabs(ref_matrix[i][j] - result_matrix[i][j]));
            // float absV = abs(ref_matrix[i][j] - result_matrix[i][j]);
            if(fabs(ref_matrix[i][j] - result_matrix[i][j]) > threshold)
            {
                check = false;
                printf("\nfirst not match at (%d, %d)", i, j);
                printf("\nref(%f) : result(%f) - diff(%f)\n", ref_matrix[i][j], result_matrix[i][j], fabs(ref_matrix[i][j] - result_matrix[i][j]));
                // break;
            }
            
        }
        // printf("\n");
        
    }
    if(abs(ref_min.value - minCElementGlobal.value) > threshold)
    {
        printf("Incorrect min value! [(ref, row, col) : (result, row, col)] - [(%f, %d, %d) : (%f, %d, %d)]\n", ref_min.value, ref_min.row, ref_min.col, minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);

    }
    else if(ref_min.row != minCElementGlobal.row)
    {
        printf("Incorrect row index! [(ref, row, col) : (result, row, col)] - [(%f, %d, %d) : (%f, %d, %d)]\n", ref_min.value, ref_min.row, ref_min.col, minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);
    }
    else if(ref_min.col != minCElementGlobal.col)
    {
        printf("Incorrect column index!\n");
    }
    else printf("Min value and indexes match!\n");

    if(check) printf("Result matrix match!\n");
    else printf("Result matrix does not match!\n");
}

//Matrix B transpose for cache optimization using cache spatial locality
void transposeMatrix(float mat[][MATRIX_SIZE])
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {   
        float temp = 0;
        for (int j = i; j < MATRIX_SIZE; j++)
        {
            // start = rdtsc();
            temp = mat[i][j];
            mat[i][j] = mat[j][i];
            mat[j][i] = temp;
            // end = rdtsc();
        }
        
    }

}

void* transposeMatrixAndMultiply(void* thread_args)
{      
    struct thread_args* curr_thread_args = thread_args;
    int minIndex;
    // struct matElement minCElementThread[curr_thread_args->row_end - curr_thread_args->row_start];
    //int minValue, minValueRow, minValeCol;
    matrixTransposeISPC(curr_thread_args->row_start, curr_thread_args->row_end, curr_thread_args->A, curr_thread_args->B, curr_thread_args->C);
    pthread_barrier_wait(&barrier);
    matrixMultiplyISPC(pthread_self(), curr_thread_args->row_start, curr_thread_args->row_end, curr_thread_args->A, curr_thread_args->B, curr_thread_args->C, minCElementThread);

    float minValue = FLT_MAX;
    for(int i = curr_thread_args->row_start; i < curr_thread_args->row_end; i++)
    {
        if(minCElementThread[i].value < minValue)
        {
            minIndex = i;
            minValue = minCElementThread[i].value;
        }
    }
    pthread_mutex_lock(&minCElement_mutex);
    if(minCElementThread[minIndex].value < minCElementGlobal.value)
    {
        minCElementGlobal.value = minCElementThread[minIndex].value;
        minCElementGlobal.row = minCElementThread[minIndex].row;
        minCElementGlobal.col = minCElementThread[minIndex].col;
    }
    pthread_mutex_unlock(&minCElement_mutex);
}


int main(int argc, char* argv[]){
    unsigned long long start, end;
    struct timeval start_time, end_time;
    double exec_time;
    minCElementGlobal.value = FLT_MAX;
    struct matElement minCElementThread[MATRIX_SIZE];
    int num_threads;

    if(argc < 2)
    {
        printf("Required arguments not given\nUsage:\nnum_threads: int");
        return 1;
    }
    num_threads = atoi(argv[1]);
    printf("num argc - %d\n", num_threads);

    pthread_barrier_init(&barrier, NULL, num_threads);
    pthread_t thread_id[num_threads];
    struct thread_args thread_args[num_threads];

    //Work decomposition
    float matrixSizeByNumThreads = (float)MATRIX_SIZE/(float)num_threads;
    float ceilFloatAvg = (ceil(matrixSizeByNumThreads) + floor(matrixSizeByNumThreads))/2.0;
    int threadWorkRows = (matrixSizeByNumThreads > ceilFloatAvg) ? ceil(matrixSizeByNumThreads) : floor(matrixSizeByNumThreads);
    int rowsDispatchPending = MATRIX_SIZE;


    float (*A)[MATRIX_SIZE] = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float (*B)[MATRIX_SIZE] = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float (*C)[MATRIX_SIZE] = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    if(!A || !B || !C) {
        printf("Memrory allocation falied");
        return 1;
    }

    srand(time(NULL));
    //Matrix initializations
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            A[i][j] = (float)rand() / (float)(RAND_MAX/10.0);
            B[i][j] = (float)rand() / (float)(RAND_MAX/10.0);
            C[i][j] = (float)0;
        }
        
    }

    #if DEBUG
        printf("\n\nPrinting A...\n");
        printMatrix(A);
        printf("\n\nPrinting B...\n");
        printMatrix(B);
    #endif

    
    gettimeofday(&start_time, NULL);
    
    for (int i = 0; i < num_threads - 1; i++)
    {   
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].row_start = i * threadWorkRows;
        thread_args[i].row_end = thread_args[i].row_start + threadWorkRows;
        // thread_args[i].minCElementThread = minCElementThread;
        rowsDispatchPending -= threadWorkRows;
        pthread_create(&thread_id[i], NULL, transposeMatrixAndMultiply, (void*)&thread_args[i]);
    }
    thread_args[num_threads - 1].A = A;
    thread_args[num_threads - 1].B = B;
    thread_args[num_threads - 1].C = C;
    thread_args[num_threads - 1].row_start = (num_threads - 1) * threadWorkRows;
    thread_args[num_threads - 1].row_end = thread_args[num_threads - 1].row_start + rowsDispatchPending;
    // thread_args[num_threads - 1].minCElementThread = minCElementThread;
    pthread_create(&thread_id[num_threads - 1], NULL, transposeMatrixAndMultiply, (void*)&thread_args[num_threads - 1]);
    
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(thread_id[i], NULL);
    }
    gettimeofday(&end_time, NULL);


    #if DEBUG
        //Print transposed matrix
        printf("\n\nPrinting transposed matrix B...\n");
        printMatrix(B);
    #endif
    
    
    
    #if DEBUG
        
        //Print matric C
        printf("\n\nPrinting C...\n");
        printMatrix(C);
    #endif

    checkMatrixResult(A, B, C);

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);

    printf("num_threads - %d\n", num_threads);

    printf("Minimim value in matrix C (value, row, column) - (%f, %d, %d)\n", minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);
    
    pthread_barrier_destroy(&barrier);

    free(A);
    free(B);
    free(C);
    
}