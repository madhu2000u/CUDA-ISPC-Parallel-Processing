NC = nvcc

SRC = cuda-2g-wsm2.cu

OUT = cuda-2-wsm2.out

2g:$(SRC)
	$(NC) $(SRC) -o $(OUT)


2grun:2g
	./$(OUT)


clean:
	rm $(OUT) $(OUT1g2m)
