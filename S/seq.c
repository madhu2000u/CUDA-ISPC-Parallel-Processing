#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MATRIX_SIZE 800

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

int main(){
    struct timeval start_time, end_time;
    double exec_time;
    // gettimeofday(&start_time, NULL);

    float A[MATRIX_SIZE][MATRIX_SIZE];
    float B[MATRIX_SIZE][MATRIX_SIZE];

    float C[MATRIX_SIZE][MATRIX_SIZE];

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

    // printf("\n\nPrinting A...\n");
    // printMatrix(A);
    // printf("\n\nPrinting B...\n");
    // printMatrix(B);

    //Matrix multiplication
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
            
        }
        
    }
    gettimeofday(&end_time, NULL);

    //Print matric C
    // printf("\n\nPrinting C...\n");
    // printMatrix(C);
    
    // gettimeofday(&end_time, NULL);
    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);
    
    
}