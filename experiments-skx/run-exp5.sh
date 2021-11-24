#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o run-exp1.out
#SBATCH -e run-exp1.e
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=munch@lnm.mw.tum.de
# Wall clock limit:
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr83te
#
## #SBATCH --switches=4@24:00:00
#SBATCH --partition=test
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#module list

#source ~/.bashrc
# lscpu

#module unload mkl mpi.intel intel
#module load intel/19.0 mkl/2019
#module load gcc/9
#module unload mpi.intel
#module load mpi.intel/2019_gcc
#module load cmake
#module load slurm_setup

module unload intel-mpi/2019-intel
module unload intel/19.0.5
module load gcc/9
module load intel-mpi/2019-gcc


pwd

mpirun -np 48 ./benchmark_03 "host" "annulus"  1 | tee exp_cuda_annulus_1.txt
mpirun -np 48 ./benchmark_03 "host" "annulus"  2 | tee exp_cuda_annulus_2.txt
mpirun -np 48 ./benchmark_03 "host" "annulus"  3 | tee exp_cuda_annulus_3.txt
mpirun -np 48 ./benchmark_03 "host" "annulus"  4 | tee exp_cuda_annulus_4.txt
mpirun -np 48 ./benchmark_03 "host" "annulus"  5 | tee exp_cuda_annulus_5.txt
mpirun -np 48 ./benchmark_03 "host" "annulus"  6 | tee exp_cuda_annulus_6.txt

