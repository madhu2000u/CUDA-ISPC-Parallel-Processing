#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>

#define MATRIX_SIZE 4096

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
    float minC = FLT_MAX;
    int row, col;

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
            A[i][j] = rand() / (float)1147654321;// / (float)(RAND_MAX/10.0);
            B[i][j] = rand() / (float)1147654321;// / (float)(RAND_MAX/10.0);
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
            if(C[i][j] < minC)
            {
                minC = C[i][j];
                row = i;
                col = j;
            }
            
        }
        
    }
    gettimeofday(&end_time, NULL);

    // //Print matric C
    // printf("\n\nPrinting C...\n");
    // printMatrix(C);
    
    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);

    printf("Minimim value in matrix C (value, row, column) - (%f, %d, %d)\n", minC, row, col);
    
    free(A);
    free(B);
    free(C);
    
}