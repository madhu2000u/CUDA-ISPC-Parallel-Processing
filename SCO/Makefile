CC = gcc

CFLAGS = -std=c99

TARGET = sco.out

SRC = sco.c

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:$(TARGET)
	./$(TARGET)

clean:
	rm $(TARGET)