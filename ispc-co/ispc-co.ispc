#include "my_ispc-common.h"

#define FLT_MAX 3.40282346638528859812e+38F

export void matrixTransposeAndMultiply(uniform float matA[][MATRIX_SIZE], uniform float matB[][MATRIX_SIZE], uniform float matC[][MATRIX_SIZE], uniform struct matElement minResultC[MATRIX_SIZE])
{   
    uniform float min[MATRIX_SIZE];
    // float temp = 0;
    foreach(i = 0 ... MATRIX_SIZE)
    {
        float temp = 0;
        for(int j = i; j < MATRIX_SIZE; j++)
        {
            temp = matB[i][j];
            matB[i][j] = matB[j][i];
            matB[j][i] = temp;
        }
    }

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
    foreach(i = 0 ... MATRIX_SIZE)  // Each SIMD lane gets assigned its own i value. so if 16 lanes in avx512 then in parllel all 16 lanes will have i from 0 to 15 respectively
    {
           // Varying(vector) variable that is different for different SIMD lanes.
        for(int j = 0; j < MATRIX_SIZE; j++)
        {
            
            for(int k = 0; k < MATRIX_SIZE; k++)
            {
                matC[i][j] += matA[i][k] * matB[j][k];
            }
            if(matC[i][j] < minEachC.value){ 
                minEachC.value = matC[i][j];
                minEachC.row = i;
                minEachC.col = j;

            }
        }
        minResultC[i] = minEachC;       // Each lane computes it's own min value of it's corrsponding row of the multiplication. So we would have MATRIX_SIZE number of min values
        
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

