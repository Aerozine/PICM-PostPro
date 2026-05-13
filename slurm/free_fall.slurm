#!/usr/bin/env bash
#SBATCH --job-name=picm-freefall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

POSTPRO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}"
source "$POSTPRO_ROOT/slurm/common.sh"
picm_slurm_init

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

BUILD_DIR="${BUILD_DIR:-$PICM_ROOT/build-release}"
DATA_DIR="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"

echo "[free_fall] PICM_ROOT:  $PICM_ROOT"
echo "[free_fall] PostPro:    $PICM_POSTPRO_ROOT"
echo "[free_fall] binary:     $BUILD_DIR/bin/PIC"
echo "[free_fall] data:       $DATA_DIR/free_fall"

cd "$PICM_POSTPRO_ROOT"
exec "$PYTHON_BIN" run_free_fall.py \
  --binary "$BUILD_DIR/bin/PIC" \
  --methods "${PICM_METHODS:-pic,flip,apic}" \
  --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --out "$DATA_DIR/free_fall" \
  --force \
  "$@"
