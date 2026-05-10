#!/bin/bash -l
#SBATCH --job-name="PICM PIC weak strong scaling"
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load releases/2021b
module load Info0939Tools

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH=""
for candidate in \
  "${PICM_POSTPRO_ROOT:-}/slurm/common.sh" \
  "$SUBMIT_DIR/common.sh" \
  "$SUBMIT_DIR/slurm/common.sh" \
  "$SUBMIT_DIR/PostPro/slurm/common.sh" \
  "$SUBMIT_DIR/PICM-PostPro/slurm/common.sh" \
  "$SUBMIT_DIR/PICM/PostPro/slurm/common.sh" \
  "$SCRIPT_DIR/common.sh"; do
  if [[ -f "$candidate" ]]; then
    COMMON_SH="$candidate"
    break
  fi
done
if [[ -z "$COMMON_SH" ]]; then
  echo "[error] cannot locate PostPro/slurm/common.sh from $SUBMIT_DIR" >&2
  exit 2
fi
source "$COMMON_SH"
picm_slurm_init

THREADS="${PICM_SCALING_THREADS:-1,2,4,8,16,32,64}"
BINDINGS="${PICM_SCALING_BINDINGS:-close,spread}"
PPC="${PICM_SCALING_PPC:-3}"
REPEATS="${PICM_REPEATS:-1}"
DATA_ROOT="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MISC_ROOT="${PICM_POSTPRO_MISC:-$DATA_ROOT/misc}"
IMG_ROOT="${PICM_POSTPRO_IMG:-$PICM_POSTPRO_ROOT/img}"
OUT="${PICM_STUDY_OUT:-$DATA_ROOT/study_pic_scaling}"
MISC_OUT="${PICM_STUDY_MISC:-$MISC_ROOT/study_pic_scaling}"
IMG_OUT="${PICM_STUDY_IMG:-$IMG_ROOT/study_pic_scaling}"
CONFIG="${PICM_SCALING_CONFIG:-$PICM_ROOT/test/PIC/extra/freeFallInWater.json}"

EXTRA_ARGS=(--skip-build --force)
[[ "${PICM_KEEP_RAW:-0}" == "1" ]] && EXTRA_ARGS+=(--keep-raw)
[[ "${PICM_SCALING_USE_SRUN:-1}" == "1" ]] && EXTRA_ARGS+=(--use-srun)
[[ "${PICM_SCALING_RUN_LOGS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-run-logs)
[[ "${PICM_SCALING_PLOTS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-plots)

echo "[study] PIC weak/strong scaling"
echo "[study] PICM_ROOT: $PICM_ROOT"
echo "[study] PostPro:   $PICM_POSTPRO_ROOT"
echo "[study] job id:    ${SLURM_JOB_ID:-local}"
echo "[study] nodes:     ${SLURM_JOB_NODELIST:-local}"
echo "[study] config:    $CONFIG"
echo "[study] threads:   $THREADS"
echo "[study] binding:   $BINDINGS"
echo "[study] ppc:       $PPC"
echo "[study] repeats:   $REPEATS"
echo "[study] csv output:$OUT"
echo "[study] misc:      $MISC_OUT"
echo "[study] img:       $IMG_OUT"

cd "$PICM_ROOT"
exec "$PYTHON_BIN" "$PICM_POSTPRO_ROOT/pic_scaling.py" \
  --config "$CONFIG" \
  --threads "$THREADS" \
  --bindings "$BINDINGS" \
  --ppc "$PPC" \
  --repeats "$REPEATS" \
  --out "$OUT" \
  --misc-dir "$MISC_OUT" \
  --img-dir "$IMG_OUT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
