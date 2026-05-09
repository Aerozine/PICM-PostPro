picm_has_source() {
  [[ -f "$1/CMakeLists.txt" && -d "$1/src" ]]
}

picm_find_root() {
  local postpro_root="$1"
  local submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
  for candidate in \
    "${PICM_ROOT:-}" \
    "$submit_dir" \
    "$submit_dir/PICM" \
    "$submit_dir/../PICM" \
    "$postpro_root/.." \
    "$postpro_root/../PICM" \
    "$(pwd)" \
    "$(pwd)/PICM"; do
    if [[ -n "$candidate" && -d "$candidate" ]] && picm_has_source "$candidate"; then
      cd "$candidate" && pwd
      return 0
    fi
  done
  return 1
}

picm_slurm_init() {
  local common_dir
  common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export PICM_POSTPRO_ROOT
  PICM_POSTPRO_ROOT="$(cd "$common_dir/.." && pwd)"

  export PICM_ROOT
  if ! PICM_ROOT="$(picm_find_root "$PICM_POSTPRO_ROOT")"; then
    echo "[error] cannot locate PICM root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}" >&2
    exit 2
  fi

  export PYTHON_BIN="${PYTHON:-python3}"
  export OMP_PLACES="${OMP_PLACES:-cores}"
  export OMP_DYNAMIC=false
  export MPLBACKEND=Agg
  export PYTHONDONTWRITEBYTECODE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/picm_matplotlib_${SLURM_JOB_ID:-local}}"
}
