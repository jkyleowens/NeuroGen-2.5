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

# Optional portable C++ inference engine. See NATIVE.md for GPU build options.
CMAKE ?= cmake
NATIVE_OPTIONS ?=
.PHONY: native native-test
native:
	$(CMAKE) -S . -B build/native -DCMAKE_BUILD_TYPE=Release $(NATIVE_OPTIONS)
	$(CMAKE) --build build/native --parallel 2

native-test: native
	ctest --test-dir build/native --output-on-failure
	$(PYTHON) -m unittest discover -s tests -p test_native.py -v
