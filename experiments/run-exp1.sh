mpirun -np  40 ./benchmark_01 annulus  7 7 1 6 0 0 "index" | tee exp1_annulus.txt
mpirun -np  40 ./benchmark_01 annulus  6 6 1 6 1 0 "index" | tee exp1_annulus_mapping.txt

mpirun -np  40 ./benchmark_01 quadrant 6 6 1 6 0 0 "index" | tee exp1_quadrant.txt
mpirun -np  40 ./benchmark_01 quadrant 5 5 1 6 1 0 "index" | tee exp1_quadrant_mapping.txt

