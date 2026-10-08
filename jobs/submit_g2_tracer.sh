#!/bin/bash
#SBATCH --job-name=g2_tracer
#SBATCH --partition=single
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=8gb
#SBATCH --output=logs/g2_tracer_%j.out
#SBATCH --error=logs/g2_tracer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gn316@uni-heidelberg.de

# ============================================================
# G2 ambient medium — M3 (passive tracer, 10yr run)
# bwUniCluster 2.0 SLURM script
# ============================================================

# --- Modules ---
module purge
module load compiler/gnu
module load mpi/openmpi
module load lib/hdf5/1.12

# --- Paths ---
ATHENA_DIR="$HOME/athena"
RUN_DIR="$ATHENA_DIR/runs/g2_2d_tracer"
INPUT="$ATHENA_DIR/inputs/g2/athinput.g2_2d_tracer"

# --- Setup ---
mkdir -p "$RUN_DIR"
mkdir -p "$ATHENA_DIR/logs"

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Tasks:      $SLURM_NTASKS"
echo "Start time: $(date)"
echo "============================================"

# --- Run ---
cd "$RUN_DIR"
mpirun -np $SLURM_NTASKS "$ATHENA_DIR/bin/athena" \
    -i "$INPUT" \
    -d "$RUN_DIR"

echo "============================================"
echo "End time: $(date)"
echo "============================================"
