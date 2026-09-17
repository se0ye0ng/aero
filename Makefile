.PHONY: setup smoke lint test pilot-phase0 audit-flir manifest-flir manifest-flir-pairs preprocess-flir prepare-flir-yolox record-flir-yolox-smoke prepare-flir-yolox-timing run-flir-yolox pilot-flir-rfs audit-antiuav300 audit-antiuav300-registration run-antiuav300-registration run-antiuav300-registration-v3 prepare-antiuav300-ir-yolox pilot-antiuav300-rfs audit-antiuav410 prepare-antiuav410 data-flir data-antiuav e1 e2 e3 e4 e5 e6 report deploy-bench verify replay clean

PY ?= python3
FLIR_ARCHIVE_ARG = $(if $(AERO_FLIR_ARCHIVE),--archive "$(AERO_FLIR_ARCHIVE)")
ANTIUAV300_ARCHIVE_ARG = $(if $(AERO_ANTIUAV300_ARCHIVE),--archive "$(AERO_ANTIUAV300_ARCHIVE)")

setup:
	$(PY) -m pip install -e ".[torch,track,detect,registration,deploy,dev]"
	pre-commit install

# Transfer check: run this first on a new machine. Needs no GPU and no dataset.
smoke: lint test
	@PYTHONPATH=src $(PY) -c "import aero_ir, aero_ir.rfs, aero_ir.data.mixing; print('import ok', aero_ir.__version__)"
	@echo "smoke ok - code transferred intact"

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

test:
	PYTHONPATH=src $(PY) -m pytest

# Data-free RFS separation check plus sensor-chain forward/backward timing.
pilot-phase0:
	PYTHONPATH=src $(PY) scripts/run_phase0_pilot.py --device auto --out experiments/phase0_pilot.json

audit-flir:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/audit_flir.py --root "$(AERO_FLIR_ROOT)" \
		--video-map "$(AERO_FLIR_ROOT)/../rgb_to_thermal_vid_map.json" $(FLIR_ARCHIVE_ARG)

manifest-flir:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/build_flir_manifest.py --root "$(AERO_FLIR_ROOT)"

manifest-flir-pairs:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/build_flir_pair_manifest.py --root "$(AERO_FLIR_ROOT)" \
		--video-map "$(AERO_FLIR_ROOT)/../rgb_to_thermal_vid_map.json"

preprocess-flir:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/fit_flir_preprocess.py --root "$(AERO_FLIR_ROOT)"

prepare-flir-yolox:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/prepare_flir_yolox.py --root "$(AERO_FLIR_ROOT)"

record-flir-yolox-smoke:
	@test -n "$(RUN_DIR)" || { echo "set RUN_DIR" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/record_flir_yolox_smoke.py --run-dir "$(RUN_DIR)"

prepare-flir-yolox-timing:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	@test -n "$(RUN_ID)" || { echo "set RUN_ID" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/run_flir_yolox.py prepare --mode timing \
		--run-id "$(RUN_ID)" --root "$(AERO_FLIR_ROOT)" \
		--max-train-iters "$(or $(TIMING_ITERS),96)" \
		--timing-warmup-iters "$(or $(TIMING_WARMUP_ITERS),16)"

run-flir-yolox:
	@test -n "$(SPEC)" || { echo "set SPEC" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/run_flir_yolox.py execute --spec "$(SPEC)"

pilot-flir-rfs:
	@test -n "$(AERO_FLIR_ROOT)" || { echo "set AERO_FLIR_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/run_flir_rfs_pilot.py --root "$(AERO_FLIR_ROOT)"

audit-antiuav300:
	@test -n "$(AERO_ANTIUAV300_ROOT)" || { echo "set AERO_ANTIUAV300_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/audit_antiuav300.py --root "$(AERO_ANTIUAV300_ROOT)" \
		$(ANTIUAV300_ARCHIVE_ARG)

audit-antiuav300-registration:
	@test -n "$(AERO_ANTIUAV300_ROOT)" || { echo "set AERO_ANTIUAV300_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/audit_antiuav300_registration.py \
		--root "$(AERO_ANTIUAV300_ROOT)"

run-antiuav300-registration:
	bash scripts/run_antiuav300_registration.sh

run-antiuav300-registration-v3:
	bash scripts/run_antiuav300_registration_v3.sh

prepare-antiuav300-ir-yolox:
	@test -n "$(AERO_ANTIUAV300_ROOT)" || { echo "set AERO_ANTIUAV300_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/prepare_antiuav300_ir_yolox.py \
		--root "$(AERO_ANTIUAV300_ROOT)"

pilot-antiuav300-rfs:
	@test -n "$(AERO_ANTIUAV300_ROOT)" || { echo "set AERO_ANTIUAV300_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/run_antiuav300_rfs_pilot.py --root "$(AERO_ANTIUAV300_ROOT)"

audit-antiuav410:
	@test -n "$(AERO_ANTIUAV410_ROOT)" || { echo "set AERO_ANTIUAV410_ROOT" >&2; exit 2; }
	@test -n "$(AERO_ANTIUAV410_ARCHIVE)" || { echo "set AERO_ANTIUAV410_ARCHIVE" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/audit_antiuav410.py --root "$(AERO_ANTIUAV410_ROOT)" \
		--archive "$(AERO_ANTIUAV410_ARCHIVE)"

prepare-antiuav410:
	@test -n "$(AERO_ANTIUAV410_ROOT)" || { echo "set AERO_ANTIUAV410_ROOT" >&2; exit 2; }
	PYTHONPATH=src $(PY) scripts/prepare_antiuav410.py --root "$(AERO_ANTIUAV410_ROOT)"

data-flir:
	bash scripts/download_flir.sh

data-antiuav:
	bash scripts/download_antiuav.sh

# E1 - transfer the published fixed-total mixing protocol to IR detection
e1:
	$(PY) -m aero_ir.cli run experiment=e1_reproduce

# E2 - the sign-flip test: identical grid with pretrained initialisation
e2:
	$(PY) -m aero_ir.cli run experiment=e2_pretrain_ablation

# E3 - mixing ratio crossed with budget mode
e3:
	$(PY) -m aero_ir.cli run experiment=e3_mixing_ratio

# E4 - curation policies at a fixed generated-sample budget
e4:
	$(PY) -m aero_ir.cli run experiment=e4_curation

# E5 - sequence-disjoint small-target external validation
e5:
	$(PY) -m aero_ir.cli run experiment=e5_small_target

# E6 - optional capacity sweep (YOLOX tiny/s/m/l)
e6:
	$(PY) -m aero_ir.cli run experiment=e6_capacity

report:
	$(PY) scripts/make_report.py --out experiments/report

deploy-bench:
	$(PY) -m aero_ir.cli deploy --export onnx --quantize int8 --bench

verify:
	$(PY) scripts/verify_run.py --run $(RUN)

replay:
	$(PY) scripts/verify_run.py --run $(RUN) --execute

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache
