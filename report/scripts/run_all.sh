#!/bin/bash
# Benchmark driver. The single-threaded runs (edgeR, pyedger with 1 thread) use one process
# per cohort in parallel, so both are timed under the same load; pyedger with 8 threads then
# runs alone. BLAS is pinned to one thread.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TASKS=${TASKS:-/scratch/users/steorra/analysis/omicverse_dev/_port_reports_data/edger_tasks}
OUT=${OUT:-/scratch/users/steorra/analysis/omicverse_dev/_port_reports_data/edger_runs}
RSCRIPT=${RSCRIPT:-Rscript}
PY=${PY:-python}
REPS=${REPS:-1}
JOBS=${JOBS:-9}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
export HERE TASKS OUT RSCRIPT PY REPS
GLOBS="full_BRCA full_COAD full_KIRC full_LIHC full_LUAD full_LUSC full_PRAD full_THCA sub*"

echo "== edgeR"; date
printf '%s\n' $GLOBS | xargs -P "$JOBS" -I{} sh -c '"$RSCRIPT" "$HERE/02_run_edger.R" "$TASKS" "$OUT/edgeR" "$REPS" "{}"'
echo "== pyedger, 1 thread"; date
printf '%s\n' $GLOBS | xargs -P "$JOBS" -I{} sh -c 'NUMBA_NUM_THREADS=1 "$PY" "$HERE/03_run_pyedger.py" "$TASKS" "$OUT/pyedger_numba1" "$REPS" "{}"'
echo "== pyedger, 8 threads (alone)"; date
NUMBA_NUM_THREADS=8 "$PY" "$HERE/03_run_pyedger.py" "$TASKS" "$OUT/pyedger_numba8" "$REPS" "full_*"
date
