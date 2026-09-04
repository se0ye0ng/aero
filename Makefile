.PHONY: setup smoke lint test data-flir data-antiuav e1 e2 e3 e4 e5 e6 report deploy-bench verify clean

PY ?= python3

setup:
	$(PY) -m pip install -e ".[torch,track,detect,deploy,dev]"
	pre-commit install

# Transfer check: run this first on a new machine. Needs no GPU and no dataset.
smoke: lint test
	@$(PY) -c "import aero_ir, aero_ir.rfs, aero_ir.data.mixing; print('import ok', aero_ir.__version__)"
	@echo "smoke ok - code transferred intact"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

test:
	pytest

data-flir:
	bash scripts/download_flir.sh

data-antiuav:
	bash scripts/download_antiuav.sh

# E1 - reproduce the reported negative result under its original conditions
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

# E5 - small, low-contrast target regime with a genuine coverage gap
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

clean:
	rm -rf outputs multirun .pytest_cache .ruff_cache
