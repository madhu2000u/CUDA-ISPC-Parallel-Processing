#include "my_ispc-common.h"

#define FLT_MAX 3.40282346638528859812e+38F

//Frustrating bug - when multithreading, each thread transposes its own part of the matrix and starts multiplication without checking if the entire transpose is done or not. that seems to be the freaking error!!!! use two different ispc functions and use pthread barries in main program.
//Tested with matrix size 4 and 2 threads

export void matrixTransposeISPC(uniform int id, uniform int row_start, uniform int row_end, uniform float (matA)[MATRIX_SIZE][MATRIX_SIZE], uniform float matB[MATRIX_SIZE][MATRIX_SIZE], uniform float matC[MATRIX_SIZE][MATRIX_SIZE])
{
    foreach(i = row_start ... row_end)
    {
        float temp = 0;
        for(int j = i; j < MATRIX_SIZE; j++)
        {
            #pragma ignore warning(perf)
            temp = matB[i][j];
            // if(id==0)print("\ntemp - i,j before moving\n%", temp);    //checking data race conditions. thread 0 takes lot of time to pring simd gang values while other procedes to finish and start multiplication without full transpose (if not barrier kept)
            // print("\nrow = %, col = %\n\n",i, j);  
            #pragma ignore warning(perf)          
            matB[i][j] = matB[j][i];                            
            // print("\nmatB[j][i] to i,j\n%", matB[i][j]);
            // print("\nrow = %, col = %\n\n",i, j);
            #pragma ignore warning(perf)
            matB[j][i] = temp;
            // print("\nmatBj[j][i] from temp\n%", matB[j][i]);
            // print("\nrow = %, col = %\n\n",i, j);
        }
    }
}


export void matrixMultiplyISPC(uniform int row_start, uniform int row_end, uniform float (matA)[MATRIX_SIZE][MATRIX_SIZE], uniform float matB[MATRIX_SIZE][MATRIX_SIZE], uniform float matC[MATRIX_SIZE][MATRIX_SIZE], uniform struct matElement minCElementThread[MATRIX_SIZE])// uniform float minVale, uniform int minValueRow, uniform int minValueCol)
{   
    

    // gi=i*N+j
    
    // uniform float sum;
    // for(uniform int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i += programCount)
    // {   
        
    //     for(uniform int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j += MATRIX_SIZE)
    //     {
    //         sum = 0;
    //         foreach(k = 0 ... MATRIX_SIZE){
    //             // int k = j + programIndex;
    //             float partial_prod = matA[i][k] * matB[j][k];
    //             sum = reduce_add(partial_prod);
    //         }

    //         matC[i][j] = sum;

    //     }
    // }
    struct matElement minEachC;
    minEachC.value = FLT_MAX;
    foreach(i = row_start ... row_end)  // Each SIMD lane gets assigned its own i value. so if 16 lanes in avx512 then in parllel all 16 lanes will have i from 0 to 15 respectively
    {
        for(int j = 0; j < MATRIX_SIZE; j++)
        {
            for(int k = 0; k < MATRIX_SIZE; k++)
            {
                #pragma ignore warning(perf)
                matC[i][j] += matA[i][k] * matB[j][k];
                // print("\nthreadId - %, row = %, col = %", threadId, i, j);
                // print("\nthreadId - %, row = %, col = % : values = % * %", threadId, i, j, matA[i][k], matB[j][k]);
                // print("\n(row, col) - (%, %) : values - (% * %)", i, j, matA[i][k], matB[j][k]);
                // if(k < MATRIX_SIZE - 1) print("\nrow = %, col = %: +", i, j);
            }
            // print("\n%-----C{%}-----{%} done", matC[i][j], i, j);
            #pragma ignore warning(perf)
            if(matC[i][j] < minEachC.value){ 
                #pragma ignore warning(perf)
                minEachC.value = matC[i][j];
                minEachC.row = i;
                minEachC.col = j;

            }
        }
        #pragma ignore warning(perf)
        minCElementThread[i] = minEachC;       // Each lane computes it's own min value of it's corrsponding row of the multiplication. So we would have MATRIX_SIZE number of min values
        // print("\nrow = % : MinThread[i] - %", i, minCElementThread[i].value);
    }

    // foreach(i = 0 ... MATRIX_SIZE)
    // {
    //     float minValueEach = min[i];
    //     minResultC = reduce_min(minValueEach);
    // }
    // minResultC = reduce_min(minEachC);


    // foreach(i = 0 ... MATRIX_SIZE)
    // {
    //     float minEachC = FLT_MAX;
        
    //     for(int j = 0; j < MATRIX_SIZE; j++)
    //     {
    //         if(matC[i][j] < minEachC)
    //         {
    //             minEachC = matC[i][j];
    //             minResultC = reduce_min(minEachC);
    //         }
    //     }
        
        
    // }
    
    

    
    
    
}

