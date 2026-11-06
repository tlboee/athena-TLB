#!/bin/bash
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=63
#SBATCH --time=02:00:00
#SBATCH --mem=90G
#SBATCH --job-name=peri_restart
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/peri_restart_%j.log
# Neustart CC01 adiabatisch (1024, 2048) bei 10 yr; Snapshots alle 0.05 yr von 17 bis 19.5 yr (Perizentrum 18 yr).
# Zeiten in s: 0.05 yr = 1.578e6, 17 yr = 5.3652e8, 19.5 yr = 6.1542e8.  output2/dt gross -> keine Zwischen-Restarts.
set -e
ATH=$HOME/athena
MODS=$(python3 -c "import ast; t = ast.parse(open('$ATH/jobs/param_study.py').read()); print([n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES'][0])")
module purge; module load $MODS; module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null || true
echo "Module: $MODS"
for SRC in ${RUNS:-ps_ref_big_x2_t50 ps_ref_big_x4_t50}; do
  DST=$ATH/runs/${SRC%_t50}_peri
  mkdir -p $DST && cd $DST
  cat > RUN_INFO.md <<INFO
# RUN_INFO: $(basename $DST)
Neustart von $SRC aus g2_cart.00001.rst (10 yr), Snapshots alle 0.05 yr von 17 bis 19.5 yr.
Binary: $ATH/bin/athena_g2_cart (Stand $(cd $ATH && git log -1 --format='%h %ad' --date=short)); Original lief mit 0c6952a2.
Abgeschickt: $(date +%F), Job $SLURM_JOB_ID
INFO
  echo "== $SRC -> $DST  $(date)"
  mpirun -np $SLURM_NTASKS $ATH/bin/athena_g2_cart -r $ATH/runs/$SRC/g2_cart.00001.rst -d . \
      output1/dt=1.578e6 output1/next_time=5.3652e8 output2/dt=3.156e10 time/tlim=6.1542e8
done
echo "fertig $(date)"
