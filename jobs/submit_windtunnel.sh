#!/bin/bash
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=63
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --job-name=windtunnel
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/windtunnel_%j.log
# Windkanal mit Kuehlung.  FAC = Lambda_fac, TLIM (optional, s) ueberschreibt tlim (fuer Kurztests).
set -e
ATH=$HOME/athena
MODS=$(python3 -c "import ast; t = ast.parse(open('$ATH/jobs/param_study.py').read()); print([n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES'][0])")
module purge; module load $MODS; module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null || true
DST=$ATH/runs/wt_L${FAC}${SUFFIX}
mkdir -p $DST && cd $DST
[ -n "$RST" ] || printf "# RUN_INFO: %s\nWindkanal, Lambda_fac = %s, Eingabe inputs/g2/athinput.g2_windtunnel, Binary-Stand %s, Job %s, %s\n" \
  "$(basename $DST)" "$FAC" "$(cd $ATH && git log -1 --format='%h %ad' --date=short)" "$SLURM_JOB_ID" "$(date +%F)" > RUN_INFO.md
if [ -n "$RST" ]; then            # Fortsetzung aus Restart-Datei im Laufordner
  echo "Fortsetzung aus $RST, Job $SLURM_JOB_ID, $(date +%F)" >> RUN_INFO.md
  mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -r $RST -d .
else
  mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -i $ATH/inputs/g2/athinput.g2_windtunnel -d . \
      problem/Lambda_fac=$FAC ${TLIM:+time/tlim=$TLIM}
fi
echo "fertig $(date)"
