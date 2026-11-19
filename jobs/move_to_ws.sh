#!/bin/bash
#SBATCH --partition=dev_cpu_il
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --job-name=move_ws
#SBATCH --output=/home/hd/hd_hd/hd_gn316/athena/runs/move_ws_%j.log
# Verschiebt Laufordner in den Workspace und ersetzt sie durch Verknuepfungen. Bricht ein Job ab, einfach neu starten:
# bereits verschobene Ordner (Verknuepfungen) werden uebersprungen, halbfertige Kopien setzt rsync fort.
WS=$(ws_find g2runs); [ -d "$WS" ] || { echo "kein Workspace g2runs"; exit 1; }
cd $HOME/athena/runs
for d in grid_ppm_20261022 ps_ref_big_x4_t50 ps_ref_big_R0.25_x4_t50 ps_ref_big_cool_x4_t50 ps_ref_big_cool_R0.25_x4_t50 \
         ps_ref_big_x4_peri d3_cc01_adiab g_chi10_rp4_R0.25_x4_t50 g_chi100_rp4_R0.25_x4_t50 g_chi1000_rp4_R0.25_x4_t50 \
         g_chi1000_rp4_R0.25_x4_t50_fine g_chi10_rp4_R1_x4_t30 g_chi1000_rp32_R0.5_x2; do
  if [ -L "$d" ]; then echo "$d: schon verschoben"; continue; fi
  [ -d "$d" ] || { echo "$d: fehlt"; continue; }
  rsync -a --delete "$d/" "$WS/$d/" || { echo "$d: rsync FEHLER, Original bleibt"; continue; }
  n1=$(find "$d" -type f | wc -l); n2=$(find "$WS/$d" -type f | wc -l)
  s1=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
  s2=$(find "$WS/$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
  if [ "$n1" = "$n2" ] && [ "$s1" = "$s2" ]; then
    rm -rf "$d" && ln -s "$WS/$d" "$d" && echo "$d: verschoben ($n1 Dateien, $((s1/1024/1024/1024)) GB)"
  else
    echo "$d: Kopie unvollstaendig ($n1/$n2 Dateien, $s1/$s2 Bytes), Original bleibt"
  fi
done
echo "fertig $(date)"
