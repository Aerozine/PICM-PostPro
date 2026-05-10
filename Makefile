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
VIDEO_DIR ?= $(POSTPRO_ROOT)/video
RELEASE_BUILD_DIR ?= $(PICM_ROOT)/build-report-release
DEBUG_BUILD_DIR ?= $(PICM_ROOT)/build-solver-debug
BUILD_JOBS ?= 32
PYTHON ?= python3
SBATCH ?= sbatch
CMAKE_ARGS ?=
EIGEN_ROOT ?=
CMAKE_COMMON_ARGS := -DUSE_GPU=OFF -DUSE_PARALLEL=ON $(if $(EIGEN_ROOT),-DCMAKE_PREFIX_PATH="$(EIGEN_ROOT)") $(CMAKE_ARGS)

POSTPRO_RUN_TEST ?= falling-block-water
POSTPRO_RUN_METHODS ?= pic,flip
POSTPRO_RUN_PPC ?= 3
POSTPRO_RUN_FLIP_COEF ?= 0
POSTPRO_RUN_THREADS ?= 1
POSTPRO_RUN_SAMPLES ?= 40
POSTPRO_RUN_OUT ?= $(DATA_DIR)/postpro_run
POSTPRO_RUN_MISC ?= $(MISC_DIR)/postpro_run
POSTPRO_RUN_IMG ?= $(IMG_DIR)/postpro_run
POSTPRO_RUN_BUILD_DIR ?= $(PICM_ROOT)/build-local-report-release
POSTPRO_RUN_BUILD_JOBS ?= 1
POSTPRO_VIDEO_WORKERS ?= 1
POSTPRO_RUN_ARGS ?=
FREE_FALL_METHODS ?= pic
FREE_FALL_THREADS ?= 1
FREE_FALL_NT ?= 300
FREE_FALL_PPC ?= 3
FREE_FALL_SAMPLING_RATE ?= 5
FREE_FALL_OUT ?= $(DATA_DIR)/study_free_fall_particles
FREE_FALL_MISC ?= $(MISC_DIR)/study_free_fall_particles
FREE_FALL_IMG ?= $(IMG_DIR)/study_free_fall_particles
FREE_FALL_ARGS ?=
VIDEO_CONFIG_ROOTS ?= test/PIC,test/FLIP,test/APIC
VIDEO_MISC ?= $(MISC_DIR)/video
VIDEO_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
VIDEO_FPS ?= 30
VIDEO_SAMPLE ?= 1
VIDEO_WIDTH ?= 1280
VIDEO_HEIGHT ?= 720
VIDEO_WORKERS ?= 1
VIDEO_BACKGROUND ?= white
VIDEO_FORCE ?= --force
VIDEO_ENCODER ?= auto
VIDEO_CRF ?= 24
VIDEO_PRESET ?= veryslow
VIDEO_ARGS ?=

STUDY_SLURM := \
	$(POSTPRO_ROOT)/slurm/study_energy.slurm \
	$(POSTPRO_ROOT)/slurm/study_vorticity.slurm \
	$(POSTPRO_ROOT)/slurm/study_ppc_impact.slurm \
	$(POSTPRO_ROOT)/slurm/study_iterative_solvers.slurm \
	$(POSTPRO_ROOT)/slurm/study_pic_scaling.slurm

.PHONY: clean build build-release build-local-release require-build sbatch postpro plot postpro-run free-fall-particles video

clean:
	find "$(PICM_ROOT)" -maxdepth 1 -type d \( -name 'build*' -o -name 'cmake-build*' \) -prune -exec rm -rf {} +
	if [[ -d "$(MISC_DIR)" ]]; then \
		find "$(MISC_DIR)" -type d -name raw -prune -exec rm -rf {} +; \
		find "$(MISC_DIR)" -type f \( -name '*.vti' -o -name '*.vtp' -o -name '*.pvd' -o -name '*.mp4' \) -delete; \
	fi
	rm -rf "$(IMG_DIR)"
	rm -rf "$(VIDEO_DIR)"
	find "$(POSTPRO_ROOT)" "$(PICM_ROOT)" -type d -name __pycache__ -prune -exec rm -rf {} +

build-release:
	cmake -S "$(PICM_ROOT)" -B "$(RELEASE_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release $(CMAKE_COMMON_ARGS)
	cmake --build "$(RELEASE_BUILD_DIR)" -j"$(BUILD_JOBS)"
	@test -x "$(RELEASE_BUILD_DIR)/bin/PIC" || { echo "[error] build finished but missing $(RELEASE_BUILD_DIR)/bin/PIC"; exit 1; }

build-local-release:
	cmake -S "$(PICM_ROOT)" -B "$(POSTPRO_RUN_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release $(CMAKE_COMMON_ARGS)
	cmake --build "$(POSTPRO_RUN_BUILD_DIR)" -j"$(POSTPRO_RUN_BUILD_JOBS)"
	@test -x "$(POSTPRO_RUN_BUILD_DIR)/bin/PIC" || { echo "[error] build finished but missing $(POSTPRO_RUN_BUILD_DIR)/bin/PIC"; exit 1; }

build: build-release
	cmake -S "$(PICM_ROOT)" -B "$(DEBUG_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Debug $(CMAKE_COMMON_ARGS)
	cmake --build "$(DEBUG_BUILD_DIR)" -j"$(BUILD_JOBS)"
	@test -x "$(DEBUG_BUILD_DIR)/bin/PIC" || { echo "[error] build finished but missing $(DEBUG_BUILD_DIR)/bin/PIC"; exit 1; }

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

postpro-run: build-local-release
	PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_DATA="$(DATA_DIR)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_IMG="$(IMG_DIR)" PICM_POSTPRO_VIDEO="$(VIDEO_DIR)" $(PYTHON) "$(POSTPRO_ROOT)/report_compare.py" \
		--analysis vorticity \
		--test "$(POSTPRO_RUN_TEST)" \
		--methods "$(POSTPRO_RUN_METHODS)" \
		--ppc "$(POSTPRO_RUN_PPC)" \
		--flip-coef-pic "$(POSTPRO_RUN_FLIP_COEF)" \
		--threads "$(POSTPRO_RUN_THREADS)" \
		--samples "$(POSTPRO_RUN_SAMPLES)" \
		--out "$(POSTPRO_RUN_OUT)" \
		--misc-dir "$(POSTPRO_RUN_MISC)" \
		--img-dir "$(POSTPRO_RUN_IMG)" \
		--build-dir "$(POSTPRO_RUN_BUILD_DIR)" \
		--skip-build \
		--force \
		--video-dir "$(VIDEO_DIR)/postpro_run" \
		--video-methods "pic,flip,apic" \
		--video-cmap "viridis" \
		--video-workers "$(POSTPRO_VIDEO_WORKERS)" \
		$(POSTPRO_RUN_ARGS)

free-fall-particles: build-local-release
	PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_DATA="$(DATA_DIR)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_IMG="$(IMG_DIR)" $(PYTHON) "$(POSTPRO_ROOT)/free_fall_particles.py" \
		--methods "$(FREE_FALL_METHODS)" \
		--ppc "$(FREE_FALL_PPC)" \
		--threads "$(FREE_FALL_THREADS)" \
		--nt "$(FREE_FALL_NT)" \
		--sampling-rate "$(FREE_FALL_SAMPLING_RATE)" \
		--out "$(FREE_FALL_OUT)" \
		--misc-dir "$(FREE_FALL_MISC)" \
		--img-dir "$(FREE_FALL_IMG)" \
		--build-dir "$(POSTPRO_RUN_BUILD_DIR)" \
		--skip-build \
		$(FREE_FALL_ARGS)

video: build-release
	PICM_ROOT="$(PICM_ROOT)" PICM_POSTPRO_MISC="$(MISC_DIR)" PICM_POSTPRO_VIDEO="$(VIDEO_DIR)" $(PYTHON) "$(POSTPRO_ROOT)/video_all.py" \
		--binary "$(RELEASE_BUILD_DIR)/bin/PIC" \
		--config-roots "$(VIDEO_CONFIG_ROOTS)" \
		--video-dir "$(VIDEO_DIR)" \
		--misc-dir "$(VIDEO_MISC)" \
		--threads "$(VIDEO_THREADS)" \
		--fps "$(VIDEO_FPS)" \
		--sample "$(VIDEO_SAMPLE)" \
		--width "$(VIDEO_WIDTH)" \
		--height "$(VIDEO_HEIGHT)" \
		--workers "$(VIDEO_WORKERS)" \
		--cmap "viridis" \
		--background "$(VIDEO_BACKGROUND)" \
		--encoder "$(VIDEO_ENCODER)" \
		--crf "$(VIDEO_CRF)" \
		--preset "$(VIDEO_PRESET)" \
		$(VIDEO_FORCE) \
		$(VIDEO_ARGS)
