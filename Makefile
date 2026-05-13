SHELL := /usr/bin/env bash
POSTPRO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Locate PICM root: try parent of PostPro, then parent/PICM
PICM_ROOT ?= $(shell \
	if [ -f "$(POSTPRO_ROOT)/../CMakeLists.txt" ] && [ -d "$(POSTPRO_ROOT)/../src" ]; then \
		cd "$(POSTPRO_ROOT)/.." && pwd; \
	elif [ -f "$(POSTPRO_ROOT)/../PICM/CMakeLists.txt" ] && [ -d "$(POSTPRO_ROOT)/../PICM/src" ]; then \
		cd "$(POSTPRO_ROOT)/../PICM" && pwd; \
	else \
		cd "$(POSTPRO_ROOT)/.." && pwd; \
	fi)

DATA_DIR        ?= $(POSTPRO_ROOT)/data
IMG_DIR         ?= $(POSTPRO_ROOT)/img
VIDEO_DIR       ?= $(POSTPRO_ROOT)/video
BUILD_DIR       ?= $(PICM_ROOT)/build-release
DEBUG_BUILD_DIR ?= $(PICM_ROOT)/build-debug
BUILD_JOBS      ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
PYTHON          ?= python3

# Thread count: respect SLURM if available, else all CPUs
THREADS ?= $(shell $(PYTHON) -c \
	"import os; v=os.environ.get('SLURM_CPUS_PER_TASK'); print(v if v else os.cpu_count() or 1)" \
	2>/dev/null || echo 1)

# Study defaults (override on command line: make run REPORT_TEST=dambreak)
REPORT_TEST      ?= falling-block-water
REPORT_METHODS   ?= pic,flip,apic
REPORT_PPC       ?= 3
REPORT_FLIP_COEF ?= 0,0.01,0.05,0.1
REPORT_ANALYSIS  ?= vorticity
PPC_PPC          ?= 1,2,3,5,8,10
FREE_FALL_METHODS ?= pic,flip,apic

.PHONY: clean build sbatch run video plot

clean:
	rm -rf "$(DATA_DIR)" "$(IMG_DIR)" "$(VIDEO_DIR)"
	find "$(PICM_ROOT)" -maxdepth 1 -type d \( -name 'build-*' -o -name 'cmake-build*' \) \
	  -exec rm -rf {} + 2>/dev/null || true
	find "$(POSTPRO_ROOT)" "$(PICM_ROOT)" -type d -name __pycache__ \
	  -exec rm -rf {} + 2>/dev/null || true

build:
	cmake -S "$(PICM_ROOT)" -B "$(BUILD_DIR)" \
	  -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF -DUSE_PARALLEL=ON
	cmake --build "$(BUILD_DIR)" -j"$(BUILD_JOBS)"
	cmake -S "$(PICM_ROOT)" -B "$(DEBUG_BUILD_DIR)" \
	  -DCMAKE_BUILD_TYPE=Debug -DUSE_GPU=OFF -DUSE_PARALLEL=ON
	cmake --build "$(DEBUG_BUILD_DIR)" -j"$(BUILD_JOBS)"

sbatch:
	@test -x "$(BUILD_DIR)/bin/PIC" || \
	  { echo "[error] run 'make build' first ($(BUILD_DIR)/bin/PIC not found)"; exit 1; }
	PICM_ROOT="$(PICM_ROOT)" \
	PICM_POSTPRO_DATA="$(DATA_DIR)" \
	PICM_POSTPRO_IMG="$(IMG_DIR)" \
	BUILD_DIR="$(BUILD_DIR)" \
	DEBUG_BUILD_DIR="$(DEBUG_BUILD_DIR)" \
	  sbatch slurm/report.slurm
	PICM_ROOT="$(PICM_ROOT)" \
	PICM_POSTPRO_DATA="$(DATA_DIR)" \
	BUILD_DIR="$(BUILD_DIR)" \
	  sbatch slurm/free_fall.slurm
	PICM_ROOT="$(PICM_ROOT)" \
	PICM_POSTPRO_DATA="$(DATA_DIR)" \
	BUILD_DIR="$(BUILD_DIR)" \
	  sbatch slurm/scaling.slurm
	PICM_ROOT="$(PICM_ROOT)" \
	PICM_POSTPRO_DATA="$(DATA_DIR)" \
	DEBUG_BUILD_DIR="$(DEBUG_BUILD_DIR)" \
	  sbatch slurm/iterative.slurm

run:
	$(PYTHON) run_report.py \
	  --binary "$(BUILD_DIR)/bin/PIC" \
	  --test "$(REPORT_TEST)" \
	  --methods "$(REPORT_METHODS)" \
	  --ppc "$(REPORT_PPC)" \
	  --analysis "$(REPORT_ANALYSIS)" \
	  --flip-coef "$(REPORT_FLIP_COEF)" \
	  --threads "$(THREADS)" \
	  --out "$(DATA_DIR)/report"
	$(PYTHON) run_report.py \
	  --binary "$(BUILD_DIR)/bin/PIC" \
	  --test "$(REPORT_TEST)" \
	  --methods pic \
	  --ppc "$(PPC_PPC)" \
	  --analysis "$(REPORT_ANALYSIS)" \
	  --threads "$(THREADS)" \
	  --out "$(DATA_DIR)/report"
	$(PYTHON) run_free_fall.py \
	  --binary "$(BUILD_DIR)/bin/PIC" \
	  --methods "$(FREE_FALL_METHODS)" \
	  --threads "$(THREADS)" \
	  --out "$(DATA_DIR)/free_fall"
	$(PYTHON) run_iterative.py \
	  --binary "$(DEBUG_BUILD_DIR)/bin/PIC" \
	  --threads 1 \
	  --out "$(DATA_DIR)/iterative"

video:
	$(PYTHON) video.py \
	  --binary "$(BUILD_DIR)/bin/PIC" \
	  --threads "$(THREADS)" \
	  --out "$(VIDEO_DIR)"

plot:
	$(PYTHON) plot.py \
	  --data "$(DATA_DIR)" \
	  --img "$(IMG_DIR)"
