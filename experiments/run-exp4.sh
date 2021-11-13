mpirun -np 40 ./benchmark_02 quadrant 7 4 1 1 | tee exp4_1_1.txt
mpirun -np 40 ./benchmark_02 quadrant 7 4 0 1 | tee exp4_0_1.txt
mpirun -np 40 ./benchmark_02 quadrant 7 4 1 0 | tee exp4_1_0.txt
mpirun -np 40 ./benchmark_02 quadrant 7 4 0 0 | tee exp4_0_0.txt
