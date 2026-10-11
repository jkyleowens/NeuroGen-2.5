PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
DURATION_MS ?= 15000

.PHONY: all check test run simulate benchmark language-doctor gui
all: check

check:
	$(PYTHON) -c "from neurogen import BiologicalNetwork; n=BiologicalNetwork(); print(n.report())"

test:
	$(PYTHON) -m unittest discover -s tests -v

run simulate:
	$(PYTHON) -m neurogen simulate --duration-ms $(DURATION_MS)

benchmark:
	$(PYTHON) benchmark_biology.py

language-doctor:
	$(PYTHON) -m neurogen language doctor

gui:
	$(PYTHON) -m neurogen gui
