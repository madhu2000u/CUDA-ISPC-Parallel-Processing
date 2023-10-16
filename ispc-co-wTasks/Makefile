CC = gcc

ISPC = ./ispc

CFLAGS = -std=c11

TARGET = ispc-co-wTasks.out

SRC = ispc-co-wTasks.c

ISPC_SRC = ispc-co-wTasks.ispc

ISPC_OBJ = ispc-co-wTasks.o

ISPC_H = ispc-co-wTasks.h

all:$(TARGET) my_ispc-common.h

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $(ISPC_OBJ) $(SRC) -o $(TARGET)

$(SRC):$(ISPC)

ispc:$(ISPC_SRC)
	$(ISPC) $(ISPC_SRC) -o $(ISPC_OBJ) -h $(ISPC_H)

run: all
	./$(TARGET)

clean:
	rm $(TARGET) $(ISPC_OBJ) $(ISPC_H)