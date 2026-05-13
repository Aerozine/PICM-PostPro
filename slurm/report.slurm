#!/usr/bin/env bash
#SBATCH --job-name=picm-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

POSTPRO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$POSTPRO_ROOT/slurm/common.sh"
picm_slurm_init

export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

BUILD_DIR="${BUILD_DIR:-$PICM_ROOT/build-release}"
DATA_DIR="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"

echo "[report] PICM_ROOT:  $PICM_ROOT"
echo "[report] PostPro:    $PICM_POSTPRO_ROOT"
echo "[report] binary:     $BUILD_DIR/bin/PIC"
echo "[report] data:       $DATA_DIR/report"

cd "$PICM_POSTPRO_ROOT"
exec "$PYTHON_BIN" run_report.py \
  --binary "$BUILD_DIR/bin/PIC" \
  --threads "${SLURM_CPUS_PER_TASK:-32}" \
  --test "${PICM_TEST:-falling-block-water}" \
  --methods "${PICM_METHODS:-pic,flip,apic}" \
  --ppc "${PICM_PPC:-3}" \
  --analysis "${PICM_ANALYSIS:-vorticity}" \
  --flip-coef "${PICM_FLIP_COEF:-0,0.01,0.05,0.1}" \
  --out "$DATA_DIR/report" \
  --force \
  "$@"
