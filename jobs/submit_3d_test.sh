#!/bin/bash
#SBATCH --job-name=g2_3d_test
#SBATCH --partition=dev_cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --output=logs/g2_3d_test_%j.out
# 3D-Test: CC01 bei 512 x 384 x 144, 2 yr; misst zone-cycles/s und Speicher
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0
module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null
cd ~/athena
mkdir -p runs/d3_cc01_test
mpirun -n $SLURM_NTASKS bin/athena_g2_cart -i inputs/g2/athinput.g2_3d_cc01 -d runs/d3_cc01_test \
  time/tlim=6.312e7 > runs/d3_cc01_test/log.txt
