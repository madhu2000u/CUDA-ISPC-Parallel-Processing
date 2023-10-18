CC = gcc

ISPC = ./ispc

CFLAGS = -std=gnu99

TARGET = pth-co-wispc.out

SRC = pth-co-wispc.c

ISPC_SRC = pth-co-wispc.ispc

ISPC_OBJ = pth-co-wispc.o

ISPC_H = pth-co-wispc.h

all:$(TARGET) my_ispc-common.h

$(TARGET):$(SRC)
	$(CC) -g $(CFLAGS) $(ISPC_OBJ) $(SRC) -o $(TARGET) -lm -pthread

$(SRC):$(ISPC)

ispc:$(ISPC_SRC)
	$(ISPC) -g $(ISPC_SRC) -o $(ISPC_OBJ) -h $(ISPC_H)


run:all
	./$(TARGET) > output.txt

clean:
	rm $(TARGET) $(ISPC_OBJ) $(ISPC_H)