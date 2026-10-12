#!/bin/bash
#SBATCH --job-name=g2_tracer
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8gb
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/logs/g2_tracer_%j.out
#SBATCH --error=/home/hd/hd_hd/hd_gn316/athena/logs/g2_tracer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gn316@uni-heidelberg.de

module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0

ATHENA_DIR="$HOME/athena"
RUN_DIR="$ATHENA_DIR/runs/g2_tracer"
INPUT="$ATHENA_DIR/inputs/g2/athinput.g2_2d_tracer"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

mpirun -np $SLURM_NTASKS "$ATHENA_DIR/bin/athena" -i "$INPUT" -d "$RUN_DIR"

echo "Done at $(date)"
echo "--- tracer check (bg=0.0, cloud=1.0) ---"
awk 'NF==10 {print $5, $10}' "$RUN_DIR"/g2_2d_tracer.block0.out1.00000.tab 2>/dev/null | sort -u | head -5
