#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define MATRIX_SIZE 2048
#define DEBUG 0
#define CHECK 0
#define NUM_THREADS 16


struct matElement
{
    float value;
    int row, col;
};

struct matElement minCElementGlobal;

struct thread_args
{   
    float (*A)[MATRIX_SIZE];
    float (*B)[MATRIX_SIZE];
    float (*C)[MATRIX_SIZE];
    int row_start, row_end;
};

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
            // float absV = abs(ref_matrix[i][j] - result_matrix[i][j]);
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

//Matrix multiplication
void transposeMatrixAndMultiply(float A[][MATRIX_SIZE], float B[][MATRIX_SIZE], float C[][MATRIX_SIZE])
{   
    
    struct matElement minCElement;
    minCElement.value = FLT_MAX;
        // #pragma omp parallel
        // {
        #pragma omp for
        for (int i = 0; i < MATRIX_SIZE; i++)
        {   
            float temp = 0;
            for (int j = i + 1; j < MATRIX_SIZE; j++)
            {
                // start = rdtsc();
                temp = B[i][j];
                B[i][j] = B[j][i];
                B[j][i] = temp;
                // end = rdtsc();
                // printf("Thread %d working on row and col - (%d, %d)\n", omp_get_thread_num(), i, j);
            } 
        }
        // }
    
    #pragma omp barrier

    #pragma omp parallel
    {   
        #pragma omp for
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
        // printf("num threads created - %d\n", omp_get_num_threads());
        //     printf("Thread %d working on row %d\n", omp_get_thread_num(), i);
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                for (int k = 0; k < MATRIX_SIZE; k++)
                {
                    // args->C[i][j] = pthread_self() ;
                    C[i][j] += A[i][k] * B[j][k];
                    // printf("Thread %d working on C[%d][%d] with current row and col - (%d, %d)\n", omp_get_thread_num(), i, j, i, j);
                }
                if(C[i][j] < minCElement.value)
                {
                    minCElement.value = C[i][j];
                    minCElement.row = i;
                    minCElement.col = j;
                }
                
            }

        }
        #pragma omp critical
            if(minCElement.value < minCElementGlobal.value)
            {
                minCElementGlobal.value = minCElement.value;
                minCElementGlobal.row = minCElement.row;
                minCElementGlobal.col = minCElement.col;
            }
        // printf("I am thread %d still active\n", omp_get_thread_num());
    }
    // printf("I am thread %d still active\n", omp_get_thread_num());
    
    
}
int main(){
    unsigned long long start, end;
    struct timeval start_time, end_time;
    double exec_time;
    minCElementGlobal.value = FLT_MAX;
    
    omp_set_num_threads(NUM_THREADS);
    
    float (*A)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*B)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));
    float (*C)[MATRIX_SIZE] = malloc(sizeof(float[MATRIX_SIZE][MATRIX_SIZE]));

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
    
    transposeMatrixAndMultiply(A, B, C);
    
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

    printf("NUM_THREADS - %d\n", NUM_THREADS);

    printf("Minimim value in matrix C (value, row, column) - (%f, %d, %d)\n", minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);
    
    free(A);
    free(B);
    free(C);
    
}