#!/bin/bash
#SBATCH --partition=dev_cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=63
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --job-name=fine_start
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/fine_start_%j.log
# Neu von t = 0 mit dem Aufruf aus manifest.json des Originallaufs; Snapshots alle 0.1 yr von t_peri - 0.5 bis t_peri + 5 yr.
# Fuer Laeufe, deren Restart-Dateien nicht mehr zur aktuellen Binary passen.  RUNS = "lauf:t_peri ..."  Ausgabe: runs/<lauf>_fine
set -e
ATH=$HOME/athena; YR=31560000
MODS=$(python3 -c "import ast; t = ast.parse(open('$ATH/jobs/param_study.py').read()); print([n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES'][0])")
module purge; module load $MODS; module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null || true
for ITEM in $RUNS; do
  SRC=${ITEM%%:*}; TP=${ITEM##*:}
  T0=$(python3 -c "print(($TP - 0.5) * $YR)"); T1=$(python3 -c "print(($TP + 5.0) * $YR)"); DT=$(python3 -c "print(0.1 * $YR)")
  CMD=$(python3 -c "import json; print(json.load(open('$ATH/runs/$SRC/manifest.json'))['commands'][0])")
  DST=$ATH/runs/${SRC}_fine; rm -rf $DST; mkdir -p $DST && cd $DST
  printf "# RUN_INFO: %s\nNeu von t = 0 (Restart-Dateien von %s passen nicht zur aktuellen Binary), Aufruf aus dessen manifest.json.\nSnapshots alle 0.1 yr von %s bis %s yr (t_peri = %s yr). Binary-Stand %s. Job %s, %s\n" \
    "$(basename $DST)" "$SRC" "$(python3 -c "print($TP - 0.5)")" "$(python3 -c "print($TP + 5)")" "$TP" \
    "$(cd $ATH && git log -1 --format='%h %ad' --date=short)" "$SLURM_JOB_ID" "$(date +%F)" > RUN_INFO.md
  echo "== $SRC -> $DST  $(date)"
  # Stufe 1: bis t_peri - 0.5 yr ohne Snapshots; final.rst mit der aktuellen Binary
  eval "$CMD output1/dt=3.156e10 output2/dt=3.156e10 time/tlim=$T0"
  mv g2_cart.final.rst stage1.rst
  # Stufe 2: Neustart mit Snapshots alle 0.1 yr (next_time ist jetzt aus der Restart-Datei bekannt)
  mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -r stage1.rst -d . \
      output1/dt=$DT output1/next_time=$T0 output2/dt=3.156e10 time/tlim=$T1
done
echo "fertig $(date)"
