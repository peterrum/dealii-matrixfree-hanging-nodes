for types in "BRANCH" "CACHES" "FLOPS_DP"
do
    for degree in 1 2 3 4 5 6
    do
        likwid-mpirun -np 40 -f -g $types -m -O ./benchmark_01_likwid annulus 6 $degree 1 0 1 1 | tee exp2_0_annulus_"$types"_"$degree".txt
        likwid-mpirun -np 40 -f -g $types -m -O ./benchmark_01_likwid annulus 6 $degree 1 1 1 1 | tee exp2_1_annulus_"$types"_"$degree".txt
        likwid-mpirun -np 40 -f -g $types -m -O ./benchmark_01_likwid annulus 6 $degree 1 1 1 0 | tee exp2_2_annulus_"$types"_"$degree".txt
    done
done

