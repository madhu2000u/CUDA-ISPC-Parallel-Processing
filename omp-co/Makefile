CC = gcc

CFLAGS = -std=gnu99 -fopenmp

TARGET = omp-co.out

SRC = omp-co.c

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:all
	./$(TARGET) $(ARGS)

clean:
	rm $(TARGET)