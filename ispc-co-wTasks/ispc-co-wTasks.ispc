#include "my_ispc-common.h"

#define FLT_MAX 3.40282346638528859812e+38F

static task void matrixTranspose(uniform float matB[][MATRIX_SIZE])
{   
    uniform float min[MATRIX_SIZE];
    uniform int row_start = (taskIndex * (MATRIX_SIZE/taskCount));
    uniform int row_end = row_start + (MATRIX_SIZE/taskCount);
    foreach(i = row_start ... row_end)
    {
        float temp = 0;
        for(int j = i; j < MATRIX_SIZE; j++)
        {   
            #pragma ignore warning(perf)
            temp = matB[i][j];
            #pragma ignore warning(perf)
            matB[i][j] = matB[j][i];
            #pragma ignore warning(perf)
            matB[j][i] = temp;
        }
    }
    
}

static task void matrixMultiply(uniform float matA[][MATRIX_SIZE], uniform float matB[][MATRIX_SIZE], uniform float matC[][MATRIX_SIZE], uniform struct matElement minResultC[MATRIX_SIZE])
{
    struct matElement minEachC;
    uniform int row_start = (taskIndex * (MATRIX_SIZE/taskCount));
    uniform int row_end = row_start + (MATRIX_SIZE/taskCount);
    minEachC.value = FLT_MAX;
    foreach(i = row_start ... row_end)  // Each SIMD lane gets assigned its own i value. so if 16 lanes in avx512 then in parllel all 16 lanes will have i from 0 to 15 respectively
    {
        for(int j = 0; j < MATRIX_SIZE; j++)
        {
            
            for(int k = 0; k < MATRIX_SIZE; k++)
            {   
                #pragma ignore warning(perf)
                matC[i][j] += matA[i][k] * matB[j][k];
            }
            #pragma ignore warning(perf)
            if(matC[i][j] < minEachC.value){ 
                #pragma ignore warning(perf)
                minEachC.value = matC[i][j];
                minEachC.row = i;
                minEachC.col = j;

            }
        }
        #pragma ignore warning(perf)
        minResultC[i] = minEachC;       // Each lane computes it's own min value of it's corrsponding row of the multiplication. So we would have MATRIX_SIZE number of min values
        
    }
}

export void matrixTransposeAndMultiply_withTasks(uniform float matA[][MATRIX_SIZE], uniform float matB[][MATRIX_SIZE], uniform float matC[][MATRIX_SIZE], uniform struct matElement minResultC[MATRIX_SIZE], uniform int tasks)
{
    launch[tasks] matrixTranspose(matB);
    sync;
    launch[tasks] matrixMultiply(matA, matB, matC, minResultC);
    sync;
}

