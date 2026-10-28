#!/bin/bash
#SBATCH --job-name=g2_cart_hr_s12
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16gb
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/logs/g2_cart_hr_s12_%j.out
#SBATCH --error=/home/hd/hd_hd/hd_gn316/athena/logs/g2_cart_hr_s12_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gn316@uni-heidelberg.de

module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0
module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null

ATHENA_DIR="$HOME/athena"
RUN_DIR="$ATHENA_DIR/runs/g2_cart_hr_s12"
INPUT="$ATHENA_DIR/inputs/g2/athinput.g2_cart_cc01_s12"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

mpirun -np $SLURM_NTASKS "$ATHENA_DIR/bin/athena_g2_cart" -i "$INPUT" -d "$RUN_DIR" mesh/nx1=1024 mesh/nx2=768 meshblock/nx1=64 meshblock/nx2=64 time/tlim=2.0356e9 output1/dt=3.156e7 || { echo "mpirun failed"; exit 1; }

echo "Done at $(date)"
echo "--- tracer check (bg=0.0, cloud=1.0) ---"
awk 'NF==10 {print $5, $10}' "$RUN_DIR"/g2_cart.block0.out1.00000.tab 2>/dev/null | sort -u | head -5
