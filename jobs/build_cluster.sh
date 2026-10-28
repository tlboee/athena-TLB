#!/bin/bash
# Athena++ auf dem Cluster sauber mit GCC bauen (NVIDIA-SDK setzt sonst OMPI_CXX=nvc++).
# Aufruf:  source jobs/build_cluster.sh          (oder: bash jobs/build_cluster.sh clean)
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0
module unload toolkit/nvidia-hpc-sdk devel/cuda 2>/dev/null
export OMPI_CXX=g++ OMPI_CC=gcc
unset CXX CC
if ! mpicxx --showme | grep -q "^g++"; then
  echo "FEHLER: mpicxx ruft nicht g++ auf:"; mpicxx --showme | cut -c1-80; return 1 2>/dev/null || exit 1
fi
cd ~/athena
[ "$1" = "clean" ] && make clean
make -j 16 2>&1 | grep -iE "error" | head -5
cp bin/athena bin/athena_g2_cart
if ldd bin/athena_g2_cart | grep -qiE "nvhpc|libnv"; then
  echo "FEHLER: Binary haengt an NVIDIA-Bibliotheken"; ldd bin/athena_g2_cart | grep -iE "nv"
else
  echo "ok: bin/athena_g2_cart mit GCC gebaut"; ldd bin/athena_g2_cart | grep -E "libstdc"
fi
