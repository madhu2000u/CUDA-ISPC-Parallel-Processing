#define FLT_MAX 3.40282346638528859812e+38F
#define MATRIX_SIZE 16
export void matrixTransposeAndMultiply(uniform float matA[][MATRIX_SIZE], uniform float matB[][MATRIX_SIZE], uniform float matC[][MATRIX_SIZE], uniform float minResultC[MATRIX_SIZE])
{
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

    
    foreach(i = 0 ... MATRIX_SIZE)  // Each SIMD lane gets assigned its own i value. so if 16 lanes in avx512 then in parllel all 16 lanes will have i from 0 to 15 respectively
    {
        float minEachC = FLT_MAX;   // Varying(vector) variable that is different for different SIMD lanes.
        for(int j = 0; j < MATRIX_SIZE; j++)
        {
            
            for(int k = 0; k < MATRIX_SIZE; k++)
            {
                matC[i][j] += matA[i][k] * matB[j][k];
            }
            if(matC[i][j] < minEachC){ 
                minEachC = matC[i][j];
                minResultC[i] = minEachC;   // Each lane computes it's own min value of it's corrsponding row of the multiplication. So we would have MATRIX_SIZE number of min values
            }
            
        }
        
    }

    
    
    
}

