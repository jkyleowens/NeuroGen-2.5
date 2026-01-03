# NeuroGen 2.0 Makefile

# Compiler settings
NVCC = nvcc
CXX = g++
CUDA_PATH ?= /opt/cuda
PYTHON_VERSION ?= $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE ?= $(shell python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYBIND11_INCLUDE ?= $(shell python3 -m pybind11 --includes 2>/dev/null || echo "")

# Architecture flags (Adjust for your GPU, e.g., sm_86 for RTX 30 series, sm_75 for T4/20 series)
ARCH_FLAGS = -arch=sm_75

# Compiler flags
NVCC_FLAGS = $(ARCH_FLAGS) -Xcompiler -fPIC -O3 -std=c++17 --expt-relaxed-constexpr -I./include -I$(CUDA_PATH)/include -I./src/persistence
CXX_FLAGS = -fPIC -O3 -std=c++17 -Wall -Wextra

# Include directories
INCLUDES = -I./include -I$(CUDA_PATH)/include -I./src/persistence

# Python binding flags (only used if pybind11 is available)
PYTHON_FLAGS = $(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE)

# Libraries
LIBS = -lcudart -lcusparse -lcublas -lcurand -lpthread

# Source files
# Engine core
ENGINE_SRCS = src/engine/NeuralEngine.cu \
              src/engine/SparseMatrix.cu \
              src/engine/kernels/LIF_Update.cu \
              src/engine/kernels/SpMV_Input.cu

# Cortical Column Architecture (V2)
CORTICAL_SRCS = src/engine/CorticalColumnV2.cu \
                src/engine/ConnectivityGenerator.cu \
                src/engine/ALIFNeuron.cu \
                src/engine/AgenticNeuron.cu \
                src/engine/AgenticSynapse.cu \
                src/engine/CorticalColumnAgentic.cu \
                src/engine/BioConnectivityGenerator.cu \
                src/engine/HebbianLearningKernel_Bio.cu

# Modules
MODULE_SRCS = src/modules/CorticalModule.cu \
              src/modules/InterModuleConnection.cpp \
              src/modules/BrainOrchestrator.cpp

# Persistence
PERSISTENCE_SRCS = src/persistence/CheckpointReader.cpp \
                   src/persistence/CheckpointWriter.cpp \
                   src/persistence/CheckpointFormat.cpp

# Main entry point
MAIN_SRC = src/main.cpp

# Interface stubs (if needed by older code, though mainly unused in Phase 2 main)
INTERFACE_SRCS = src/interfaces/TokenEmbedding.cpp \
                 src/interfaces/OutputDecoder.cpp \
                 src/interfaces/GPUDecoder.cu \
                 src/interfaces/TrainingLoop.cpp

# Object files
ENGINE_OBJS = $(ENGINE_SRCS:.cu=.o)
MODULE_OBJS = $(MODULE_SRCS:.cu=.o) $(MODULE_SRCS:.cpp=.o)
PERSISTENCE_OBJS = $(PERSISTENCE_SRCS:.cpp=.o)
INTERFACE_OBJS = $(INTERFACE_SRCS:.cpp=.o)
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# Output binaries
TARGET = neurogen_sim
PYTHON_MODULE = bin/libneurogen.so

# Rules
all: $(TARGET) python_bindings

# Check if pybind11 is available for python bindings
python_bindings:
	@if [ -n "$(PYBIND11_INCLUDE)" ]; then \
		echo "✓ pybind11 found, building Python bindings..."; \
		$(MAKE) $(PYTHON_MODULE); \
	else \
		echo "⚠️  pybind11 not found. Skipping Python bindings."; \
		echo "   Install with: pip install pybind11"; \
		echo "   Building basic shared library instead..."; \
		$(MAKE) $(PYTHON_MODULE)_basic; \
	fi

# Clean up the object list for linking to avoid duplication
LINK_OBJS = src/engine/NeuralEngine.o \
            src/engine/SparseMatrix.o \
            src/engine/kernels/LIF_Update.o \
            src/engine/kernels/SpMV_Input.o \
            src/engine/CorticalColumnV2.o \
            src/engine/ConnectivityGenerator.o \
            src/engine/ALIFNeuron.o \
            src/engine/AgenticNeuron.o \
            src/engine/AgenticSynapse.o \
            src/engine/CorticalColumnAgentic.o \
            src/engine/BioConnectivityGenerator.o \
            src/engine/HebbianLearningKernel_Bio.o \
            src/modules/CorticalModule.o \
            src/modules/InterModuleConnection.o \
            src/modules/BrainOrchestrator.o \
            src/persistence/CheckpointReader.o \
            src/persistence/CheckpointWriter.o \
            src/persistence/CheckpointFormat.o \
            src/interfaces/TokenEmbedding.o \
            src/interfaces/OutputDecoder.o \
            src/interfaces/GPUDecoder.o \
            src/interfaces/TrainingLoop.o

# Build executable
$(TARGET): $(LINK_OBJS) src/main.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA_PATH)/lib64 $(LIBS)

# Build Python module with pybind11 bindings
$(PYTHON_MODULE): $(LINK_OBJS) src/python/neurogen_bindings.o
	@mkdir -p bin
	$(CXX) $(CXX_FLAGS) -shared -o $@ $^ -L$(CUDA_PATH)/lib64 $(LIBS)

# Build basic shared library (without python bindings)
$(PYTHON_MODULE)_basic: $(LINK_OBJS)
	@mkdir -p bin
	$(CXX) $(CXX_FLAGS) -shared -o $(PYTHON_MODULE) $^ -L$(CUDA_PATH)/lib64 $(LIBS)

# Compile Python bindings
src/python/neurogen_bindings.o: src/python/neurogen_bindings.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(PYTHON_FLAGS) -c $< -o $@

# Compile CUDA source
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(PYTHON_MODULE)
	rm -f src/engine/*.o src/engine/kernels/*.o src/modules/*.o src/persistence/*.o src/interfaces/*.o src/python/*.o src/*.o
	rm -f test_cortical_column

# Test cortical column architecture
test_cortical: src/engine/test_cortical_column.o src/engine/CorticalColumnV2.o src/engine/ConnectivityGenerator.o src/engine/ALIFNeuron.o
	$(CXX) $(CXX_FLAGS) -o test_cortical_column $^ -L$(CUDA_PATH)/lib64 $(LIBS)
	@echo "Run with: ./test_cortical_column"

run: $(TARGET)
	./$(TARGET)

train: $(PYTHON_MODULE)
	python3 train_slimpajama.py

.PHONY: all clean run train python_bindings test_cortical
