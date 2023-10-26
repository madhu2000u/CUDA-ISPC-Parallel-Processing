#include <iostream>
#include <mpi.h>
#include <sys/time.h>

#define MATRIX_SIZE 4096
#define NoW 2                                   //Network of Workstations

#define MCHECK(call) \
{ int result; \
    result = call; \
    if(result != MPI_SUCCESS) \
    {\
        fprintf(stderr, "Call "#call" returned error code %i \n", result); \
        MPI_Abort(MPI_COMM_WORLD, result); \
    }\
}

typedef struct
{
    float value;
    int16_t row, col;

} matElement; 

matElement* runCuda(int, int, float*, float*);

void matrixInit1(float *a, float *b)
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            a[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            b[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            // c[i * MATRIX_SIZE + j] = (float)0;
        }
        
    }
}


int main(int argc, char* argv[])
{   
    struct timeval start_time, end_time;
    double exec_time;
    MPI_Status recvStatus;
    int rank;
    int tag = 100;
    int rowSizePerWorkstation = MATRIX_SIZE / NoW;

    float *h_a, *h_b;

    float *h_a_r1, *h_b_r1;

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    size_t mpiBuffSize = (size / 2) / sizeof(float);            //per workstation

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype MPI_matElement;
    MPI_Datatype type[3] = {MPI_FLOAT, MPI_INT, MPI_INT};
    int blocklen[3] = {1, 1, 1};
    MPI_Aint disp[3];
    disp[0] = 0;
    disp[1] = sizeof(int16_t);
    disp[2] = sizeof(int16_t) + sizeof(int16_t);

    MPI_Type_create_struct(3, blocklen, disp, type, &MPI_matElement);
    MPI_Type_commit(&MPI_matElement);
    
    if(rank == 0)   //main
    {   
        matElement *minElementsFromNoW[NoW];
        matElement final[2];
        h_a = (float*)malloc(size);
        h_b = (float*)malloc(size);
        
        minElementsFromNoW[1] = (matElement*) malloc(2 * sizeof(matElement));

        matrixInit1(h_a, h_b);
        gettimeofday(&start_time, NULL);
        MCHECK(MPI_Send(h_a + mpiBuffSize , mpiBuffSize, MPI_FLOAT, 1, tag, MPI_COMM_WORLD));
        MCHECK(MPI_Send(h_b , (size / sizeof(float)), MPI_FLOAT, 1, tag, MPI_COMM_WORLD));              //no need to multiply by sizeof(float) as mpi_send implicitly determines it and if you do it will give seg fault
        // printf("sending done\n");
        minElementsFromNoW[0] = runCuda(rank, rowSizePerWorkstation, h_a, h_b);
        // printf("waiting to recienve min elements\n");
        MCHECK(MPI_Recv(minElementsFromNoW[1], 2, MPI_matElement, 1, tag, MPI_COMM_WORLD, &recvStatus));
        // printf("recieved min elements\n");
        // printf("rank 0 %d\n", (minElementsFromNoW[1]+1)->row);

        float minVal = __FLT_MAX__;
        int minIndex_j = 0;
        int minIndex_k = 0;
        for (int i = 0; i < NoW; i++)
        {
            for (int j = 0; j < NoW; j++)
            {
                for(int k = 0; k < 2; k++)
                {    
                    if((minElementsFromNoW[j] + k)->value < minVal)
                    {
                        minVal = (minElementsFromNoW[j] + k)->value;
                        minIndex_j = j;
                        minIndex_k = k;
                    }
                }
            }

            final[i] = *(minElementsFromNoW[minIndex_j] + minIndex_k);
            minVal = __FLT_MAX__;
            (minElementsFromNoW[minIndex_j] + minIndex_k)->value = __FLT_MAX__;
        }
        gettimeofday(&end_time, NULL);
        
        exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;
        std::cout<<"Matrix size - "<<MATRIX_SIZE<<std::endl;
        std::cout<<"Total Execution time by host of rank 0 - "<<exec_time<<std::endl;
        std::cout<<"Final min value 1 (val, row, col) - ("<<final[0].value<<", "<<final[0].row<<", "<<final[0].col<<"), "<<"printing from rank "<<rank<<std::endl;

        std::cout<<"Final min value 2 (val, row, col) - ("<<final[1].value<<", "<<final[1].row<<", "<<final[1].col<<"), "<<"printing from rank "<<rank<<std::endl;

    }
    else if(rank == 1)
    {   
        matElement *minElementsFromNoW;
        // printf("i am rank 1\n");
        h_a_r1 = (float*)malloc(rowSizePerWorkstation * MATRIX_SIZE * sizeof(float));
        h_b_r1 = (float*)malloc(size);
        MCHECK (MPI_Recv(h_a_r1, mpiBuffSize, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &recvStatus));
        MCHECK (MPI_Recv(h_b_r1, (size / sizeof(float)), MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &recvStatus));
        // printf("sending recved\n");
        minElementsFromNoW = runCuda(rank, rowSizePerWorkstation, h_a_r1, h_b_r1);
        // printf("size of minElementsFromNoW %lu\n", sizeof(matElement));
        MCHECK(MPI_Send(minElementsFromNoW, 2, MPI_matElement, 0, tag, MPI_COMM_WORLD)); 
        // printf("rank 1 %f\n", minElementsFromNoW[1].value);
    }

    MPI_Finalize();
    
}