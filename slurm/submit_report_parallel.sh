#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH="$SCRIPT_DIR/common.sh"
if [[ ! -f "$COMMON_SH" ]]; then
  echo "[error] cannot locate $COMMON_SH" >&2
  exit 2
fi
source "$COMMON_SH"
picm_slurm_init

SLURM_FILE="$SCRIPT_DIR/report_single.slurm"
TESTS="${PICM_REPORT_TESTS:-von-karman,dambreak,vases-communicants}"
METHODS="${PICM_REPORT_METHODS:-pic,flip,apic}"
PPCS="${PICM_REPORT_PPC:-3,5}"
FLIP_COEFS="${PICM_FLIP_COEF_PIC:-0,0.01,0.05,0.1}"
DATA_ROOT="${PICM_POSTPRO_DATA:-$PICM_POSTPRO_ROOT/data}"
MISC_ROOT="${PICM_POSTPRO_MISC:-$DATA_ROOT/misc}"
IMG_ROOT="${PICM_POSTPRO_IMG:-$PICM_POSTPRO_ROOT/img}"

submit_one() {
  local test_name="$1"
  local method="$2"
  local ppc="$3"
  local coef="$4"
  shift 4
  local job_name="picm-${test_name}-${method}-p${ppc}"
  local export_vars="ALL,PICM_ROOT=$PICM_ROOT,PICM_POSTPRO_ROOT=$PICM_POSTPRO_ROOT,PICM_POSTPRO_DATA=$DATA_ROOT,PICM_POSTPRO_MISC=$MISC_ROOT,PICM_POSTPRO_IMG=$IMG_ROOT,PICM_REPORT_TEST=$test_name,PICM_REPORT_METHOD=$method,PICM_REPORT_PPC=$ppc"

  if [[ "$method" == "flip" ]]; then
    job_name="${job_name}-c${coef//./p}"
    export_vars="$export_vars,PICM_FLIP_COEF_PIC=$coef"
  fi

  sbatch --job-name="$job_name" --export="$export_vars" "$SLURM_FILE" "$@"
}

IFS=',' read -r -a test_list <<< "$TESTS"
IFS=',' read -r -a method_list <<< "$METHODS"
IFS=',' read -r -a ppc_list <<< "$PPCS"
IFS=',' read -r -a coef_list <<< "$FLIP_COEFS"

echo "[submit] PICM_ROOT:  $PICM_ROOT"
echo "[submit] PostPro:    $PICM_POSTPRO_ROOT"
echo "[submit] data:       $DATA_ROOT"
echo "[submit] misc:       $MISC_ROOT"
echo "[submit] img:        $IMG_ROOT"
echo "[submit] tests:      $TESTS"
echo "[submit] methods:    $METHODS"
echo "[submit] ppc:        $PPCS"
echo "[submit] flip coeff: $FLIP_COEFS"

cd "$PICM_ROOT"

for test_name in "${test_list[@]}"; do
  for method in "${method_list[@]}"; do
    for ppc in "${ppc_list[@]}"; do
      if [[ "$method" == "flip" ]]; then
        for coef in "${coef_list[@]}"; do
          submit_one "$test_name" "$method" "$ppc" "$coef" "$@"
        done
      else
        submit_one "$test_name" "$method" "$ppc" "" "$@"
      fi
    done
  done
done
