CC = gcc

ISPC = ./ispc

CFLAGS = -std=c99

TARGET = ispc-co.out

SRC = ispc-co.c

ISPC_SRC = ispc-co.ispc

ISPC_OBJ = ispc-co.o

ISPC_H = ispc-co.h

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(ISPC_OBJ) $(SRC) -o $(TARGET)

$(SRC):$(ISPC)

ispc:$(ISPC_SRC)
	$(ISPC) $(ISPC_SRC) -o $(ISPC_OBJ) -h $(ISPC_H)

run:$(TARGET)
	./$(TARGET)

clean:
	rm $(TARGET) $(ISPC_OBJ) $(ISPC_H)