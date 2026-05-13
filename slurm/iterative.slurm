#!/usr/bin/env bash
#SBATCH --job-name=picm-iterative
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

POSTPRO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}"
source "$POSTPRO_ROOT/slurm/common.sh"
picm_slurm_init

DEBUG_BUILD_DIR="${DEBUG_BUILD_DIR:-$PICM_ROOT/build-debug}"
DATA_DIR="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"

echo "[iterative] PICM_ROOT:      $PICM_ROOT"
echo "[iterative] PostPro:        $PICM_POSTPRO_ROOT"
echo "[iterative] debug binary:   $DEBUG_BUILD_DIR/bin/PIC"
echo "[iterative] data:           $DATA_DIR/iterative"

cd "$PICM_POSTPRO_ROOT"
exec "$PYTHON_BIN" run_iterative.py \
  --binary "$DEBUG_BUILD_DIR/bin/PIC" \
  --threads "${SLURM_CPUS_PER_TASK:-32}" \
  --out "$DATA_DIR/iterative" \
  --force \
  "$@"
