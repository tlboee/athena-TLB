#!/bin/bash
#SBATCH --job-name=g2_3d_cc01
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/g2_3d_cc01_%j.out
# 3D-Lauf CC01, adiabatisch (Max, 30.10.): 512 x 384 x 144, z = +-2e16 cm, PLM, CFL 0.3, bis 50 yr.
# Ausgabe: Mittelebene jedes Jahr (output1, x3_slice = 0), Restart alle 10 yr (~3 GB), History.
# Verlaengern: im Lauf-Ordner  mpirun ... -r g2_cart.final.rst -d . time/tlim=<neue Endzeit in s>
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0
module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null
cd ~/athena
mkdir -p runs/d3_cc01_adiab
cd runs/d3_cc01_adiab
mpirun -n $SLURM_NTASKS ~/athena/bin/athena_g2_cart -i ~/athena/inputs/g2/athinput.g2_3d_cc01 -d . \
  time/xorder=2 time/cfl_number=0.3 output1/dt=3.156e7 output2/dt=3.156e8 time/tlim=1.578e9 > log.txt
