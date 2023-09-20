CC = gcc

CFLAGS = -std=c99

TARGET = seq.out

SRC = seq.c

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:$(TARGET)
	./$(TARGET)

clean:
	rm $(TARGET)