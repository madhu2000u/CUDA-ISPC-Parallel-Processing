CC = gcc

ISPC = ispc

CFLAGS = -std=gnu99 -fopenmp

TARGET = omp-co-wispc.out

SRC = omp-co-wispc.c

ISPC_SRC = omp-co-wispc.ispc

ISPC_OBJ = omp-co-wispc.o

ISPC_H = omp-co-wispc.h

all:$(TARGET) my_ispc-common.h

$(TARGET):$(SRC) $(ISPC_OBJ)
	$(CC) $(CFLAGS) $(ISPC_OBJ) $(SRC) -o $(TARGET) -lm

$(ISPC_OBJ):$(ISPC_SRC)
	$(ISPC) $(ISPC_SRC) -o $(ISPC_OBJ) -h $(ISPC_H)


run:all
	./$(TARGET) $(ARGS)

clean:
	rm $(TARGET) $(ISPC_OBJ) $(ISPC_H)
