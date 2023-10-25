MPIC = mpic++ -g

CUDAC = nvcc

MPI_SRC = main-mpi.cpp

MPI_OBJ = main-mpi.o

CUDA_SRC = cuda-mpi.cu

CUDA_OBJ = cuda-mpi.o

TARGET = cuda-mpi.out

HOSTNAME = $(shell hostname)

EXEC_DIR = ~/ese565workingdir/pa/cuda-mpi/

all:$(TARGET)

$(TARGET):$(CUDA_OBJ) $(MPI_OBJ)
ifeq ($(HOSTNAME), bizon)
	$(MPIC) $^ -lcudart -lstdc++ -L/usr/local/cuda-12.0/targets/x86_64-linux/lib -o $@
	
else ifeq ($(HOSTNAME), lion.ece.stonybrook.edu)
	$(MPIC) $^ -lcudart -lstdc++ -L/usr/lib/nvidia-cuda-toolkit/lib64 -o $@
	
endif
	echo current host is $(HOSTNAME)

$(CUDA_OBJ):$(CUDA_SRC)
	$(CUDAC) -c $^ -o $@

$(MPI_OBJ):$(MPI_SRC)
	$(MPIC) -c $^ -o $@

mpi:$(MPI_SRC)
	$(MPIC) 

sendremote:$(CUDA_SRC) $(MPI_SRC) Makefile
ifeq ($(HOSTNAME), bizon)
	scp -o "ForwardAgent yes" $^ lion:$(EXEC_DIR)
	ssh lion "cd $(EXEC_DIR) && make all"

else
	scp -o "ForwardAgent yes" $^ bizon:$(EXEC_DIR)
	ssh bizon "cd $(EXEC_DIR) && make all"

endif

run:all
	./$(TARGET)

mpirun:sendremote all
	mpirun --host bizon,lion:1 $(EXEC_DIR)/$(TARGET)

# mpirun -np 1 --host lion $(EXEC_DIR)/$(TARGET) : -np 1  --host bizon $(EXEC_DIR)/$(TARGET)

# xterm -e gdb

clean:
	rm $(CUDA_OBJ) $(MPI_OBJ) $(TARGET)