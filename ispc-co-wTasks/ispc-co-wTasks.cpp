#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <stdbool.h>
#include "ispc-co-wTasks.h"
#include "my_ispc-common.h"

using namespace ispc;

void printMatrix(float mat[][MATRIX_SIZE])
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
           printf("%f    ", mat[i][j]);
            
        }
        printf("\n");
        
    }
}

void checkMatrixResult(float A[][MATRIX_SIZE], float B[][MATRIX_SIZE], float ref_matrix[][MATRIX_SIZE], float result_matrix[][MATRIX_SIZE], struct matElement minValueC)
{
    bool check = true;
    double threshold = 1.0E-2;
    struct matElement ref_min;
    ref_min.value = FLT_MAX;

    // Reference Matrix multiplication
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
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
    if(abs(ref_min.value - minValueC.value) > threshold || ref_min.row != minValueC.row || ref_min.col != minValueC.col) check = false;

    if(check) printf("Result matrix match!\n");
    else printf("Result matrix does not match or min value and index do not match!\n");
}

unsigned long long rdtsc()
{
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

int main(int argc, char* argv[]){
    unsigned long long start, end;
    int matrixSize, tasks;
    struct timeval start_time, end_time;
    double exec_time;
    struct matElement minResultC[MATRIX_SIZE];
    struct matElement finalMin;

    if(argc < 2)
    {
        printf("Required arguments not given\nUsage:\ntasks: int\n\nIf you are using makefile then Usage is:\nmake run ARGS=16, where ARGS is the number threads/cores/tasks\n");
        return 1;
    }
    tasks = atoi(argv[1]);

    float (*A)[MATRIX_SIZE] = (float (*)[MATRIX_SIZE]) malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*B)[MATRIX_SIZE] = (float (*)[MATRIX_SIZE]) malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*C)[MATRIX_SIZE] = (float (*)[MATRIX_SIZE]) malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*ref_matrix)[MATRIX_SIZE] = (float (*)[MATRIX_SIZE]) malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));

    if(!A || !B || !C) {
        printf("Memrory allocation falied");
        return 1;
    }

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

    //Matrix B transpose for cache optimization using cache spatial locality
    gettimeofday(&start_time, NULL);

    matrixTransposeAndMultiply_withTasks(A, B, C, minResultC, tasks);

    finalMin.value = minResultC[0].value;
    for(int i = 1; i < MATRIX_SIZE; i++)
    {
        if(minResultC[i].value < finalMin.value) finalMin = minResultC[i];
    }

    gettimeofday(&end_time, NULL);
    
    #if DEBUG
        //Print transposed matrix
        printf("\n\nPrinting transposed matrix B...\n");
        printMatrix(B);
    #endif
    

    

    
    gettimeofday(&end_time, NULL);
    // checkMatrixResult(A, B, ref_matrix, C, finalMin);
    #if DEBUG
        //Print matric C
        printf("\n\nPrinting C...\n");
        printMatrix(C);

        printf("\n\nPrinting ref_matrix...\n");
        printMatrix(ref_matrix);       
    #endif

    

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);

    printf("Num tasks - %d\n", tasks);

    printf("\n\nPrinting minVal...(%f, %d, %d)\n", finalMin.value, finalMin.row, finalMin.col);
    
    free(A);
    free(B);
    free(C);
    
}