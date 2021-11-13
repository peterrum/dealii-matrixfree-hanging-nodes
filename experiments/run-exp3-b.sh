mpirun -np  40 ./benchmark_01 annulus  7 7 1 6 0 0 "group" | tee exp3b_annulus.txt

mpirun -np  40 ./benchmark_01 quadrant 6 6 1 6 0 0 "group" | tee exp3b_quadrant.txt

