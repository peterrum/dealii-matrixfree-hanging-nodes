mpirun -np  40 ./benchmark_01 annulus  7 7 1 6 0 1 "sorted" | tee exp3a_annulus.txt

mpirun -np  40 ./benchmark_01 quadrant 6 6 1 6 0 1 "sorted" | tee exp3a_quadrant.txt

