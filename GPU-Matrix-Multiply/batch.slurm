#!/bin/bash
# Account name
#SBATCH --account=cse260
# Job name,  will be displayed on the showq command
#SBATCH --job-name=MMPY-CUDA
# Filename for standard output 
# At end of job, it is in directory from which sbatch was invoked
#SBATCH -o MMPY-CUDA.o%j
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p CLUSTER
#  The requested wall clock job time limit in HH:MM:SS
#  Your job will end when it exceeds this time limit
#SBATCH -t 00:05:00 

export OMP_NUM_THREADS=16
export KMP_AFFINITY="granularity=core,scatter"

module load cuda9


# Print out the environment
printenv


date

# Run
./mmpy -n 256 -r 10 -h
./mmpy -n 512 -r 10 -h
./mmpy -n 767 -r 10 -h
./mmpy -n 768 -r 10 -h
./mmpy -n 769 -r 10 -h
./mmpy -n 1023 -r 10 -h
./mmpy -n 1024 -r 10 -h
./mmpy -n 1025 -r 10 -h
./mmpy -n 1279 -r 10 -h
./mmpy -n 1280 -r 10 -h
./mmpy -n 1281 -r 10 -h
./mmpy -n 1535 -r 10 -h
./mmpy -n 1536 -r 10 -h
./mmpy -n 1537 -r 10 -h
./mmpy -n 1279 -r 10 -h
./mmpy -n 1280 -r 10 -h
./mmpy -n 1281 -r 10 -h
./mmpy -n 1535 -r 10 -h
./mmpy -n 1536 -r 10 -h
./mmpy -n 1537 -r 10 -h
./mmpy -n 1791 -r 10 -h
./mmpy -n 1792 -r 10 -h
./mmpy -n 1793 -r 10 -h
./mmpy -n 2047 -r 10 -h
./mmpy -n 2048 -r 10 -h
./mmpy -n 2049 -r 10 -h
echo ">>> Job Ends"

date


