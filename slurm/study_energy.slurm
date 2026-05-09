#!/usr/bin/env bash
#SBATCH --job-name=picm-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

TEST_NAME="${PICM_REPORT_TEST:-falling-block-water}"
METHODS="${PICM_REPORT_METHODS:-pic,flip,apic}"
PPC_VALUE="${PICM_PPC_VALUE:-3}"
FLIP_COEFS="${PICM_FLIP_COEF_PIC:-0,0.05,0.1}"
THREADS="${PICM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
REPEATS="${PICM_REPEATS:-1}"
SAMPLES="${PICM_SAMPLES:-40}"
DATA_ROOT="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MISC_ROOT="${PICM_POSTPRO_MISC:-$DATA_ROOT/misc}"
IMG_ROOT="${PICM_POSTPRO_IMG:-$PICM_POSTPRO_ROOT/img}"
OUT="${PICM_STUDY_OUT:-$DATA_ROOT/study_energy}"
MISC_OUT="${PICM_STUDY_MISC:-$MISC_ROOT/study_energy}"
IMG_OUT="${PICM_STUDY_IMG:-$IMG_ROOT/study_energy}"

EXTRA_ARGS=(--skip-build)
[[ "${PICM_KEEP_RAW:-0}" == "1" ]] && EXTRA_ARGS+=(--keep-raw)
[[ "${PICM_RUN_LOGS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-run-logs)
[[ "${PICM_PLOTS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-plots)

echo "[study] Energy comparison"
echo "[study] PICM_ROOT:  $PICM_ROOT"
echo "[study] PostPro:    $PICM_POSTPRO_ROOT"
echo "[study] test:       $TEST_NAME"
echo "[study] methods:    $METHODS"
echo "[study] ppc:        $PPC_VALUE"
echo "[study] coefPic:    $FLIP_COEFS"
echo "[study] threads:    $THREADS"
echo "[study] csv output: $OUT"
echo "[study] misc:       $MISC_OUT"
echo "[study] img:        $IMG_OUT"

cd "$PICM_ROOT"
exec "$PYTHON_BIN" "$PICM_POSTPRO_ROOT/report_compare.py" \
  --analysis energy \
  --test "$TEST_NAME" \
  --methods "$METHODS" \
  --ppc "$PPC_VALUE" \
  --flip-coef-pic "$FLIP_COEFS" \
  --threads "$THREADS" \
  --repeats "$REPEATS" \
  --samples "$SAMPLES" \
  --out "$OUT" \
  --misc-dir "$MISC_OUT" \
  --img-dir "$IMG_OUT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
