NC = nvcc

SRC1g1m = cuda-1g-wsm1.cu

SRC1g2m = cuda-1g-wsm2.cu

OUT1g1m = cuda-1g-wsm1.out

OUT1g2m = cuda-1g-wsm2.out

1m:$(SRC1g1m)
	$(NC) $(SRC1g1m) -o $(OUT1g1m)

2m:$(SRC1g2m)
	$(NC) $(SRC1g2m) -o $(OUT1g2m)

1mrun:1m
	./$(OUT1g1m) > output.txt

2mrun:2m
	./$(OUT1g2m) > output2m.txt

clean:
	rm $(OUT1g1m) $(OUT1g2m)
