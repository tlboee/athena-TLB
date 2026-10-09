#!/bin/bash
#SBATCH --job-name=g2_tracer_test
#SBATCH --partition=single
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4gb
#SBATCH --output=logs/g2_tracer_test_%j.out
#SBATCH --error=logs/g2_tracer_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gn316@uni-heidelberg.de

# ============================================================
# G2 tracer validation test (short, 4 MPI ranks, tlim=1yr)
# Checks: rho ∝ 1/r, r0=0 background, r0=1 cloud
# ============================================================

module purge
module load compiler/gnu
module load mpi/openmpi
module load lib/hdf5/1.12

ATHENA_DIR="$HOME/athena"
RUN_DIR="$ATHENA_DIR/runs/g2_tracer_test"
INPUT="$ATHENA_DIR/inputs/g2/athinput.g2_2d_tracer"

mkdir -p "$RUN_DIR"
mkdir -p "$ATHENA_DIR/logs"

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Tasks:      $SLURM_NTASKS"
echo "Start time: $(date)"
echo "============================================"

cd "$RUN_DIR"
mpirun -np $SLURM_NTASKS "$ATHENA_DIR/bin/athena" \
    -i "$INPUT" \
    -d "$RUN_DIR" \
    tlim=3.156e7

echo "============================================"
echo "End time: $(date)"
echo "============================================"

# Quick sanity check on output
echo "--- Sanity check: first 3 background cells r0 (should be 0.0) ---"
head -5 "$RUN_DIR"/g2_2d_tracer.block0.out1.00000.tab 2>/dev/null | awk '{print $5, $10}'
echo "--- Cloud cells r0 (should be 1.0) ---"
awk 'NF==10 && $10+0 > 0.5 {print $2, $4, $5, $10; exit}' \
    "$RUN_DIR"/g2_2d_tracer.block*.out1.00000.tab 2>/dev/null | head -2
