if [[ $1 = "0" ]] ; then
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 7 4 1 0 1 1 0
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 7 4 1 1 1 1 0
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 7 4 1 1 1 0 0
fi

if [[ $1 = "1" ]] ; then
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 6 4 1 0 1 1 1
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 6 4 1 1 1 1 1
    likwid-mpirun -np 40 -f -g MEM -m -O ./benchmark_01_likwid annulus 6 4 1 1 1 0 1
fi

if [[ $1 = "2" ]] ; then
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 7 4 1 0 1 1 0
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 7 4 1 1 1 1 0
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 7 4 1 1 1 0 0
fi

if [[ $1 = "3" ]] ; then
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 6 4 1 0 1 1 1
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 6 4 1 1 1 1 1
    likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./benchmark_01_likwid annulus 6 4 1 1 1 0 1
fi
