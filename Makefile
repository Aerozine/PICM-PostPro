SHELL := /usr/bin/env bash

POSTPRO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PICM_ROOT ?= $(shell \
	if [ -f "$(POSTPRO_ROOT)/../CMakeLists.txt" ] && [ -d "$(POSTPRO_ROOT)/../src" ]; then \
		cd "$(POSTPRO_ROOT)/.." && pwd; \
	elif [ -f "$(POSTPRO_ROOT)/../PICM/CMakeLists.txt" ] && [ -d "$(POSTPRO_ROOT)/../PICM/src" ]; then \
		cd "$(POSTPRO_ROOT)/../PICM" && pwd; \
	else \
		cd "$(CURDIR)" && pwd; \
	fi)
DATA_DIR ?= $(POSTPRO_ROOT)/data
MISC_DIR ?= $(DATA_DIR)/misc
IMG_DIR ?= $(POSTPRO_ROOT)/img
RELEASE_BUILD_DIR ?= $(PICM_ROOT)/build-report-release
DEBUG_BUILD_DIR ?= $(PICM_ROOT)/build-solver-debug
BUILD_JOBS ?= 32
PYTHON ?= python3
SBATCH ?= sbatch

STUDY_SLURM := \
	$(POSTPRO_ROOT)/slurm/study_energy.slurm \
	$(POSTPRO_ROOT)/slurm/study_vorticity.slurm \
	$(POSTPRO_ROOT)/slurm/study_ppc_impact.slurm \
	$(POSTPRO_ROOT)/slurm/study_iterative_solvers.slurm \
	$(POSTPRO_ROOT)/slurm/study_pic_scaling.slurm

.PHONY: clean build require-build sbatch postpro plot

clean:
	find "$(PICM_ROOT)" -maxdepth 1 -type d \( -name 'build*' -o -name 'cmake-build*' \) -prune -exec rm -rf {} +
	if [[ -d "$(MISC_DIR)" ]]; then \
		find "$(MISC_DIR)" -type d -name raw -prune -exec rm -rf {} +; \
		find "$(MISC_DIR)" -type f \( -name '*.vti' -o -name '*.vtp' -o -name '*.pvd' -o -name '*.mp4' \) -delete; \
	fi
	rm -rf "$(IMG_DIR)"
	find "$(POSTPRO_ROOT)" "$(PICM_ROOT)" -type d -name __pycache__ -prune -exec rm -rf {} +

build:
	cmake -S "$(PICM_ROOT)" -B "$(RELEASE_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF -DUSE_PARALLEL=ON
	cmake --build "$(RELEASE_BUILD_DIR)" -j"$(BUILD_JOBS)"
	cmake -S "$(PICM_ROOT)" -B "$(DEBUG_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Debug -DUSE_GPU=OFF -DUSE_PARALLEL=ON
	cmake --build "$(DEBUG_BUILD_DIR)" -j"$(BUILD_JOBS)"

require-build:
	@test -x "$(RELEASE_BUILD_DIR)/bin/PIC" || { echo "[error] missing $(RELEASE_BUILD_DIR)/bin/PIC; run 'make -C $(POSTPRO_ROOT) build' before sbatch"; exit 1; }
	@test -x "$(DEBUG_BUILD_DIR)/bin/PIC" || { echo "[error] missing $(DEBUG_BUILD_DIR)/bin/PIC; run 'make -C $(POSTPRO_ROOT) build' before sbatch"; exit 1; }

sbatch: require-build
	for slurm_file in $(STUDY_SLURM); do \
		echo "[sbatch] $$slurm_file"; \
		PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_ROOT="$(POSTPRO_ROOT)" PICM_POSTPRO_DATA="$(DATA_DIR)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_IMG="$(IMG_DIR)" $(SBATCH) "$$slurm_file"; \
	done

postpro:
	PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_DATA="$(DATA_DIR)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_IMG="$(IMG_DIR)" $(PYTHON) "$(POSTPRO_ROOT)/plot_all.py" --data "$(DATA_DIR)" --img "$(IMG_DIR)" --postpro-only

plot:
	PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_DATA="$(DATA_DIR)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_IMG="$(IMG_DIR)" $(PYTHON) "$(POSTPRO_ROOT)/plot_all.py" --data "$(DATA_DIR)" --img "$(IMG_DIR)"
