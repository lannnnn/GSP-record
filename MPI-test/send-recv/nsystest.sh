#!/bin/bash
#SBATCH -A paj2305
#SBATCH -N 1
#SBATCH -t 240
#SBATCH -p dc-gpu
#SBATCH -G 4

srun --nodes=1 --time=00:10:00 --gres=gpu:4 --ntasks-per-node=2 ./sendRecv 100000
# srun --nodes=2 --time=00:10:00 --gres=gpu:4 --ntasks-per-node=4 --account=paj2305 --partition=dc-gpu mpiexec -np 8 ncu  --target-processes all --export output.report --import-source=yes --page raw --set full ./sendRecv
