#!/usr/bin/env bash
#SBATCH --job-name=picm-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

POSTPRO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}"
source "$POSTPRO_ROOT/slurm/common.sh"
picm_slurm_init

# Let run_scaling.py control OMP_PROC_BIND per run
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC=false

BUILD_DIR="${BUILD_DIR:-$PICM_ROOT/build-release}"
DATA_DIR="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MAX_THREADS="${SLURM_CPUS_PER_TASK:-64}"

echo "[scaling] PICM_ROOT:  $PICM_ROOT"
echo "[scaling] PostPro:    $PICM_POSTPRO_ROOT"
echo "[scaling] binary:     $BUILD_DIR/bin/PIC"
echo "[scaling] data:       $DATA_DIR/scaling"
echo "[scaling] max threads: $MAX_THREADS"

cd "$PICM_POSTPRO_ROOT"
exec "$PYTHON_BIN" run_scaling.py \
  --binary "$BUILD_DIR/bin/PIC" \
  --threads "1,2,4,8,16,32,$(echo $MAX_THREADS)" \
  --bindings "close,spread" \
  --out "$DATA_DIR/scaling" \
  --force \
  "$@"
