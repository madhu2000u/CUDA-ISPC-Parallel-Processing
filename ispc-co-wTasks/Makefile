CC = g++

ISPC = ./ispc

CFLAGS = -std=gnu++11

TARGET = ispc-co-wTasks.out

SRC = ispc-co-wTasks.cpp

ISPC_SRC = ispc-co-wTasks.ispc

ISPC_OBJ = ispc-co-wTasks.o

ISPC_H = ispc-co-wTasks.h

ISPC_TASK_SRC = tasksys.cpp

ISPC_TASK_OBJ = tasksys.o

all:$(TARGET) my_ispc-common.h

$(TARGET):$(ISPC_TASK_OBJ) $(ISPC_OBJ) $(SRC)
	$(CC) $^ -o $@ -lpthread

$(ISPC_TASK_OBJ):$(ISPC_TASK_SRC)
	$(CC) -c $(CFLAGS) $^ -o $@

# $(ISPC_OBJ):ispc

$(ISPC_OBJ):$(ISPC_SRC)
	$(ISPC) $(ISPC_SRC) -o $(ISPC_OBJ) -h $(ISPC_H)

run: all
	./$(TARGET) $(ARGS)

clean:
	rm $(TARGET) $(ISPC_OBJ) $(ISPC_H)
