LU:LU.c
	gcc -std=c11 -mavx2 -mfma -fopenmp -O3 $^ -o $@ 
