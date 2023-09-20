#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MATRIX_SIZE 2000
#define DEBUG 0

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

int main(){
    unsigned long long start, end;
    struct timeval start_time, end_time;
    double exec_time;

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
            A[i][j] = (float)rand();
            B[i][j] = (float)rand();
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

    for (int i = 0; i < MATRIX_SIZE; i++)
    {   
        float temp = 0;
        for (int j = i; j < MATRIX_SIZE; j++)
        {
            // start = rdtsc();
            temp = B[i][j];
            B[i][j] = B[j][i];
            B[j][i] = temp;
            // end = rdtsc();
        }
        
    }
    
    #if DEBUG
        //Print transposed matrix
        printf("\n\nPrinting transposed matrix B...\n");
        printMatrix(B);
    #endif
    

    //Matrix multiplication

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                C[i][j] += A[i][k] * B[j][k];
            }
            
        }
        
    }
    gettimeofday(&end_time, NULL);

    #if DEBUG
        //Print matric C
        printf("\n\nPrinting C...\n");
        printMatrix(C);
    #endif
    
    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);
    
    free(A);
    free(B);
    free(C);
    
}