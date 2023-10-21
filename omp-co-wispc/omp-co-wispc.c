#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include "my_ispc-common.h"
#include "omp-co-wispc.h"


struct matElement minCElementGlobal;
struct matElement minCElementThread[MATRIX_SIZE];


void printMatrix(float mat[][MATRIX_SIZE]){
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

void checkMatrixResult(float A[][MATRIX_SIZE], float B[][MATRIX_SIZE], float result_matrix[][MATRIX_SIZE])
{
    bool check = true;
    double threshold = 1.0E-2;
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
            if(abs(ref_matrix[i][j] - result_matrix[i][j]) > threshold)
            {
                check = false;
                break;
            }
            
        }
        
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

int main(int argc, char* argv[]){
    unsigned long long start, end;
    struct timeval start_time, end_time;
    double exec_time;
    minCElementGlobal.value = FLT_MAX;
    int row_start, row_end;

    int num_threads;

    if(argc < 2)
    {
        printf("Required arguments not given\nUsage:\nnum_threads: int");
        return 1;
    }
    num_threads = atoi(argv[1]);
    
    omp_set_num_threads(num_threads);
    
    float (*A)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*B)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*C)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    if(!A || !B || !C) {
        printf("Memrory allocation falied");
        return 1;
    }

    // srand(time(NULL));
    //Matrix initializations
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            A[i][j] = rand() / (float)1147654321;
            B[i][j] = rand() / (float)1147654321;
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
    
    #pragma omp parallel private(row_start, row_end)
    {   //if(omp_get_thread_num() == 0) printf("num threads - %d\n", omp_get_num_threads());
        float matrixSizeByNumThreads = (float)MATRIX_SIZE/(float)num_threads;
        float ceilFloatAvg = (ceil(matrixSizeByNumThreads) + floor(matrixSizeByNumThreads))/2.0;
        int threadWorkRows = (matrixSizeByNumThreads > ceilFloatAvg) ? ceil(matrixSizeByNumThreads) : floor(matrixSizeByNumThreads);

        float minValue = FLT_MAX;
        int minIndex;
        row_start = omp_get_thread_num() * threadWorkRows;
        row_end = (omp_get_thread_num() == num_threads - 1) ? MATRIX_SIZE : row_start + threadWorkRows;

        matrixTransposeISPC(omp_get_thread_num(), row_start, row_end, A, B, C);
        #pragma omp barrier
        matrixMultiplyISPC(row_start, row_end, A, B, C, minCElementThread);

        for(int i = row_start; i < row_end; i++)
        {
            if(minCElementThread[i].value < minValue)
            {
                minIndex = i;
                minValue = minCElementThread[i].value;
            }
        }

        #pragma omp critical
        {
            if(minCElementThread[minIndex].value < minCElementGlobal.value)
            {
                minCElementGlobal.value = minCElementThread[minIndex].value;
                minCElementGlobal.row = minCElementThread[minIndex].row;
                minCElementGlobal.col = minCElementThread[minIndex].col;
            }      
        }      
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

    #if CHECK
        checkMatrixResult(A, B, C);
    #endif
    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);

    printf("num_threads - %d\n", num_threads);

    printf("Minimim value in matrix C (value, row, column) - (%f, %d, %d)\n", minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);
    
    free(A);
    free(B);
    free(C);
    
}