CC = gcc

CFLAGS = -std=gnu99

TARGET = pth-co.out

SRC = pth-co.c

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm -pthread

run:all
	./$(TARGET) $(ARGS)

clean:
	rm $(TARGET)