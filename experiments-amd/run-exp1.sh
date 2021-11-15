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

module unload intel-mpi/2019-intel
module unload intel/19.0.5
module load gcc/9
module load intel-mpi/2019-gcc


pwd

array=($(ls run-exp1-a-*.json))
mpirun -np  48 ./benchmark_01 json "${array[@]}" | tee exp1_annulus.txt

array=($(ls run-exp1-c-*.json))
mpirun -np  48 ./benchmark_01 json "${array[@]}" | tee exp1_quadrant.txt
