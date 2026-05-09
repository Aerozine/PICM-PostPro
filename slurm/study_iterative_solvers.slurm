#!/usr/bin/env bash
#SBATCH --job-name=picm-iters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

CONFIG="${PICM_SOLVER_CONFIG:-$PICM_ROOT/test/PIC/dambreak.json}"
SOLVERS="${PICM_CPU_SOLVERS:-jacobi,gauss_seidel,red_black_gauss_seidel,miccg0,cg}"
TOLERANCES="${PICM_SOLVER_TOLERANCES:-1e-2}"
THREADS="${PICM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
REPEATS="${PICM_REPEATS:-1}"
DATA_ROOT="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MISC_ROOT="${PICM_POSTPRO_MISC:-$DATA_ROOT/misc}"
IMG_ROOT="${PICM_POSTPRO_IMG:-$PICM_POSTPRO_ROOT/img}"
OUT="${PICM_STUDY_OUT:-$DATA_ROOT/study_iterative_solvers}"
MISC_OUT="${PICM_STUDY_MISC:-$MISC_ROOT/study_iterative_solvers}"
IMG_OUT="${PICM_STUDY_IMG:-$IMG_ROOT/study_iterative_solvers}"

EXTRA_ARGS=(--skip-build --force)
[[ "${PICM_KEEP_RAW:-0}" == "1" ]] && EXTRA_ARGS+=(--keep-raw)
[[ "${PICM_RUN_LOGS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-run-logs)
[[ "${PICM_PLOTS:-0}" == "0" ]] && EXTRA_ARGS+=(--no-plots)
[[ -n "${PICM_SOLVER_NT:-}" ]] && EXTRA_ARGS+=(--nt "$PICM_SOLVER_NT")
[[ -n "${PICM_SOLVER_PPC:-}" ]] && EXTRA_ARGS+=(--ppc "$PICM_SOLVER_PPC")
[[ -n "${PICM_SOLVER_MAX_ITERATIONS:-}" ]] && EXTRA_ARGS+=(--max-iterations "$PICM_SOLVER_MAX_ITERATIONS")

echo "[study] Iterative pressure solvers"
echo "[study] PICM_ROOT:  $PICM_ROOT"
echo "[study] PostPro:    $PICM_POSTPRO_ROOT"
echo "[study] config:     $CONFIG"
echo "[study] solvers:    $SOLVERS"
echo "[study] tolerances: $TOLERANCES"
echo "[study] threads:    $THREADS"
echo "[study] repeats:    $REPEATS"
echo "[study] csv output: $OUT"
echo "[study] misc:       $MISC_OUT"
echo "[study] img:        $IMG_OUT"
echo "[study] binary:     prebuilt Debug CPU OpenMP"

cd "$PICM_ROOT"
exec "$PYTHON_BIN" "$PICM_POSTPRO_ROOT/solver_iterations.py" \
  --config "$CONFIG" \
  --solvers "$SOLVERS" \
  --tolerances "$TOLERANCES" \
  --threads "$THREADS" \
  --repeats "$REPEATS" \
  --out "$OUT" \
  --misc-dir "$MISC_OUT" \
  --img-dir "$IMG_OUT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
