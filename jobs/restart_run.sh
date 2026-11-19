#!/bin/bash
#SBATCH --partition=dev_cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=63
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --job-name=restart
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/restart_%j.log
# Fortsetzung eines Laufs aus seiner letzten Restart-Datei.  Aufruf: sbatch --export=ALL,RUN=<ordner> jobs/restart_run.sh
set -e
ATH=$HOME/athena
MODS=$(python3 -c "import ast; t = ast.parse(open('$ATH/jobs/param_study.py').read()); print([n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES'][0])")
module purge; module load $MODS; module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null || true
cd $ATH/runs/$RUN
RST=${RST:-$( [ -f g2_cart.final.rst ] && echo g2_cart.final.rst || ls -t g2_cart.*.rst | head -1 )}
echo "Fortsetzung aus $RST, Job $SLURM_JOB_ID, $(date +%F), EXTRA: $EXTRA" >> RUN_INFO.md
mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -r $RST -d . $EXTRA
echo "fertig $(date)"
