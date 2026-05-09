#!/usr/bin/env bash
#SBATCH --job-name=picm-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

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

export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

TEST_NAME="${PICM_REPORT_TEST:?set PICM_REPORT_TEST, e.g. von-karman}"
METHOD="${PICM_REPORT_METHOD:?set PICM_REPORT_METHOD, e.g. pic, flip, or apic}"
PPC="${PICM_REPORT_PPC:-3}"
FLIP_COEF="${PICM_FLIP_COEF_PIC:-0.05}"
THREADS="${PICM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
REPEATS="${PICM_REPEATS:-1}"
SAMPLES="${PICM_SAMPLES:-40}"
DATA_ROOT="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MISC_ROOT="${PICM_POSTPRO_MISC:-$DATA_ROOT/misc}"
IMG_ROOT="${PICM_POSTPRO_IMG:-$PICM_POSTPRO_ROOT/img}"
OUT="${PICM_REPORT_OUT:-$DATA_ROOT/report_comparisons}"
MISC_OUT="${PICM_REPORT_MISC:-$MISC_ROOT/report_comparisons}"
IMG_OUT="${PICM_REPORT_IMG:-$IMG_ROOT/report_comparisons}"

EXTRA_ARGS=(--skip-build --force)
[[ "${PICM_KEEP_RAW:-0}" == "1" ]] && EXTRA_ARGS+=(--keep-raw)
[[ "${PICM_RUN_LOGS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-run-logs)
[[ "${PICM_PLOTS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-plots)

echo "[slurm] single report comparison"
echo "[slurm] PICM_ROOT:  $PICM_ROOT"
echo "[slurm] PostPro:    $PICM_POSTPRO_ROOT"
echo "[slurm] test:       $TEST_NAME"
echo "[slurm] method:     $METHOD"
echo "[slurm] ppc:        $PPC"
echo "[slurm] coefPic:    $FLIP_COEF"
echo "[slurm] threads:    $THREADS"
echo "[slurm] repeats:    $REPEATS"
echo "[slurm] csv output: $OUT"
echo "[slurm] misc:       $MISC_OUT"
echo "[slurm] img:        $IMG_OUT"
echo "[slurm] job id:     ${SLURM_JOB_ID:-local}"

cd "$PICM_ROOT"
exec "$PYTHON_BIN" "$PICM_POSTPRO_ROOT/report_compare.py" \
  --test "$TEST_NAME" \
  --methods "$METHOD" \
  --ppc "$PPC" \
  --flip-coef-pic "$FLIP_COEF" \
  --threads "$THREADS" \
  --repeats "$REPEATS" \
  --samples "$SAMPLES" \
  --out "$OUT" \
  --misc-dir "$MISC_OUT" \
  --img-dir "$IMG_OUT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
