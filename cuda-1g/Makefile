NC = nvcc

SRC = cuda-1g-wsm1.cu

OUT = cuda-1g-wsm1.out

1g:$(SRC)
	$(NC) $(SRC) -o $(OUT)

1grun:1g
	./$(OUT) > output.txt

clean:
	rm $(OUT)
