#!/bin/bash
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=63
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=fine_restart
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/fine_restart_%j.log
# Neustart aus g2_cart.00001.rst (10 yr) mit Snapshots alle 0.1 yr von t_peri - 0.5 bis t_peri + 5 yr.
# RUNS = "lauf:t_peri lauf:t_peri ..."   Ausgabe: runs/<lauf>_fine
set -e
ATH=$HOME/athena; YR=31560000
MODS=$(python3 -c "import ast; t = ast.parse(open('$ATH/jobs/param_study.py').read()); print([n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES'][0])")
module purge; module load $MODS; module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null || true
for ITEM in $RUNS; do
  SRC=${ITEM%%:*}; TP=${ITEM##*:}
  T0=$(python3 -c "print(($TP - 0.5) * $YR)"); T1=$(python3 -c "print(($TP + 5.0) * $YR)"); DT=$(python3 -c "print(0.1 * $YR)")
  DST=$ATH/runs/${SRC}_fine; mkdir -p $DST && cd $DST
  printf "# RUN_INFO: %s\nNeustart von %s aus g2_cart.00001.rst, Snapshots alle 0.1 yr von %s bis %s yr (t_peri = %s yr).\nBinary: %s (Stand %s). Job %s, %s\n" \
    "$(basename $DST)" "$SRC" "$(python3 -c "print($TP - 0.5)")" "$(python3 -c "print($TP + 5)")" "$TP" \
    "$ATH/bin/athena_g2_cart" "$(cd $ATH && git log -1 --format='%h %ad' --date=short)" "$SLURM_JOB_ID" "$(date +%F)" > RUN_INFO.md
  echo "== $SRC -> $DST  $(date)"
  mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -r $ATH/runs/$SRC/g2_cart.00001.rst -d . \
      output1/dt=$DT output1/next_time=$T0 output2/dt=3.156e10 time/tlim=$T1
done
echo "fertig $(date)"
