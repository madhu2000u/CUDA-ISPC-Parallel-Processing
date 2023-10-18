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

#define NUM_THREADS 32

pthread_mutex_t minCElement_mutex = PTHREAD_MUTEX_INITIALIZER;

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
            //printf("\t%f\t", fabs(ref_matrix[i][j] - result_matrix[i][j]));
            // float absV = abs(ref_matrix[i][j] - result_matrix[i][j]);
            if(fabs(ref_matrix[i][j] - result_matrix[i][j]) > threshold)
            {
                check = false;
                //printf("\nfirst not match at (%d, %d)", i, j);
                //printf("\nref(%f) : result(%f)\n", ref_matrix[i][j], result_matrix[i][j]);
                break;
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
    matrixTransposeAndMultiply(curr_thread_args->row_start, curr_thread_args->row_end, curr_thread_args->A, curr_thread_args->B, curr_thread_args->C, minCElementThread);

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

//Matrix multiplication
// void* transposeMatrixAndMultiply(void* curr_thread_args)
// {   
//     struct thread_args* args = curr_thread_args;
//     struct matElement minCElement;
//     minCElement.value = FLT_MAX;

//     for (int i = args->row_start; i < args->row_end; i++)
//     {   
//         float temp = 0;
//         for (int j = i + 1; j < MATRIX_SIZE; j++)
//         {
//             // start = rdtsc();
//             temp = args->B[i][j];
//             args->B[i][j] = args->B[j][i];
//             args->B[j][i] = temp;
//             // end = rdtsc();
//         }
        
//     }

//     //TODO barrier here
//     pthread_barrier_wait(&barrier);

//     for (int i = args->row_start; i < args->row_end; i++)
//     {
//         for (int j = 0; j < MATRIX_SIZE; j++)
//         {
//             for (int k = 0; k < MATRIX_SIZE; k++)
//             {
//                 // args->C[i][j] = pthread_self() ;
//                 args->C[i][j] += args->A[i][k] * args->B[j][k];
//             }
//             if(args->C[i][j] < minCElement.value)
//             {
//                 minCElement.value = args->C[i][j];
//                 minCElement.row = i;
//                 minCElement.col = j;
//             }
            
//         }
        
//     }
//     pthread_mutex_lock(&minCElement_mutex);
//     if(minCElement.value < minCElementGlobal.value)
//     {
//         minCElementGlobal.value = minCElement.value;
//         minCElementGlobal.row = minCElement.row;
//         minCElementGlobal.col = minCElement.col;
//     }
//     pthread_mutex_unlock(&minCElement_mutex);
    
// }


int main(){
    unsigned long long start, end;
    struct timeval start_time, end_time;
    double exec_time;
    minCElementGlobal.value = FLT_MAX;
    struct matElement minCElementThread[MATRIX_SIZE];


    pthread_t thread_id[NUM_THREADS];
    struct thread_args thread_args[NUM_THREADS];

    //Work decomposition
    float matrixSizeByNumThreads = (float)MATRIX_SIZE/(float)NUM_THREADS;
    float ceilFloatAvg = (ceil(matrixSizeByNumThreads) + floor(matrixSizeByNumThreads))/2.0;
    int threadWorkRows = (matrixSizeByNumThreads > ceilFloatAvg) ? ceil(matrixSizeByNumThreads) : floor(matrixSizeByNumThreads);
    int rowsDispatchPending = MATRIX_SIZE;


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
    // float valuesA[8][8] = {
    //     {2.832760, 3.793120, 9.574671, 5.295732, 7.831551, 0.150374, 4.652997, 9.813563},
    //     {3.162790, 1.058754, 6.093533, 6.824106, 2.250762, 9.186578, 1.429508, 1.960541},
    //     {5.724546, 5.430489, 1.585739, 3.719144, 6.881629, 0.672626, 7.088305, 1.433614},
    //     {4.315629, 6.741438, 1.274757, 5.307220, 6.705943, 5.501635, 2.910164, 2.397719},
    //     {3.215838, 3.933928, 8.554430, 4.218867, 1.522590, 6.970685, 5.377682, 8.861990},
    //     {0.896669, 0.102472, 9.168971, 5.696945, 9.903490, 4.010366, 8.660756, 1.448365},
    //     {4.236294, 6.678172, 3.035366, 8.226785, 5.788532, 3.168851, 3.601771, 0.859524},
    //     {7.634268, 6.669865, 6.371688, 2.678154, 3.565389, 9.088833, 3.096606, 4.102741}
    // };

    // float valuesB[8][8] = {
    //     {7.308180, 3.829439, 5.753661, 2.492368, 7.828209, 5.603428, 5.382293, 0.961996},
    //     {1.168221, 6.490343, 5.299218, 0.409161, 4.801577, 2.171426, 2.790725, 6.803239},
    //     {5.832077, 8.782389, 0.726222, 3.565544, 7.653072, 5.409692, 0.897039, 2.366810},
    //     {8.419926, 4.898059, 9.417290, 3.525519, 5.436060, 5.874915, 4.557956, 6.275179},
    //     {6.155407, 3.140399, 3.869518, 4.493798, 4.891493, 1.198580, 5.197471, 6.688520},
    //     {1.255336, 1.976116, 1.534626, 8.135451, 8.610894, 2.813655, 1.166214, 2.226146},
    //     {3.218787, 8.416366, 0.486189, 7.462177, 8.404300, 6.408085, 2.030840, 0.537199},
    //     {9.898988, 4.262268, 0.251095, 8.634710, 9.693310, 1.876595, 4.498439, 7.332900}
    // };

    // for (int i = 0; i < 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         A[i][j] = valuesA[i][j];
    //         B[i][j] = valuesB[i][j];
    //     }
    // }

    
    gettimeofday(&start_time, NULL);
    
    for (int i = 0; i < NUM_THREADS - 1; i++)
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
    thread_args[NUM_THREADS - 1].A = A;
    thread_args[NUM_THREADS - 1].B = B;
    thread_args[NUM_THREADS - 1].C = C;
    thread_args[NUM_THREADS - 1].row_start = (NUM_THREADS - 1) * threadWorkRows;
    thread_args[NUM_THREADS - 1].row_end = thread_args[NUM_THREADS - 1].row_start + rowsDispatchPending;
    // thread_args[NUM_THREADS - 1].minCElementThread = minCElementThread;
    pthread_create(&thread_id[NUM_THREADS - 1], NULL, transposeMatrixAndMultiply, (void*)&thread_args[NUM_THREADS - 1]);
    
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread_id[i], NULL);
    }
    gettimeofday(&end_time, NULL);


    #if DEBUG
        printf("\n\nPrinting A...\n");
        printMatrix(A);
        printf("\n\nPrinting B...\n");
        printMatrix(B);
        //Print transposed matrix
        printf("\n\nPrinting transposed matrix B...\n");
        printMatrix(B);
    #endif
    
    
    
    #if DEBUG
        
        //Print matric C
        printf("\n\nPrinting C...\n");
        printMatrix(C);
    #endif

    // checkMatrixResult(A, B, C);

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    printf("Execution time - %f\n", exec_time);
    
    printf("Matrix size - %d\n", MATRIX_SIZE);

    printf("NUM_THREADS - %d\n", NUM_THREADS);

    printf("Minimim value in matrix C (value, row, column) - (%f, %d, %d)\n", minCElementGlobal.value, minCElementGlobal.row, minCElementGlobal.col);
    
    free(A);
    free(B);
    free(C);
    
}