#ifndef NETWORK_CUDA_H
#define NETWORK_CUDA_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

// NeuroGen Framework includes
#include "engine/Network.h"
#include "engine/NetworkConfig.h"
#include "engine/LearningState.h"
#include "engine/GPUNeuralStructures.h"
#include "engine/LearningStateKernels.cuh"
#include "engine/KernelLaunchWrappers.cuh"

// Forward declarations
// Note: BrainOrchestrator is in modules/, not engine/
// NetworkCUDA operates independently in the modular architecture

/**
 * @brief GPU-accelerated neural network with persistent learning capabilities
 * 
 * This class provides CUDA-accelerated neural computation with full support for
 * persistent learning states, memory consolidation, inter-module learning,
 * and brain-inspired neural dynamics. In the modular architecture, each
 * CorticalModule contains its own NetworkCUDA instance.
 * 
 * Key Features:
 * - GPU-accelerated neural simulation with CUDA kernels
 * - Persistent learning state management on GPU
 * - Memory consolidation during downtime
 * - Inter-module connection learning
 * - Efficient GPU-CPU state synchronization
 * - Multi-stream processing for parallel modules
 * - Memory pool management for optimal performance
 * - Real-time performance monitoring
 */
class NetworkCUDA {
public:
    // ========================================================================
    // CUDA CONFIGURATION STRUCTURES
    // ========================================================================
    
    /**
     * @brief CUDA-specific configuration parameters
     */
    struct CUDAConfig {
        // Device management
        int device_id = 0;
        bool enable_unified_memory = false;
        bool enable_peer_access = true;
        size_t memory_pool_size_mb = 2048;
        
        // Kernel execution parameters
        int block_size = 256;
        int max_grid_size = 65535;
        int shared_memory_kb = 48;
        bool enable_cooperative_groups = true;
        
        // Multi-stream processing
        int num_compute_streams = 4;
        int num_memory_streams = 2;
        bool enable_stream_priorities = true;
        
        // Memory management
        bool enable_memory_pool = true;
        float memory_growth_factor = 1.5f;
        size_t max_cached_memory_mb = 4096;
        
        // Performance optimization
        bool enable_tensor_cores = true;
        bool enable_mixed_precision = false;
        bool enable_graph_capture = false;
        
        // Learning-specific settings
        bool enable_learning_state_gpu = true;
        bool enable_consolidation_gpu = true;
        size_t learning_buffer_size_mb = 512;
        
        // Debugging and profiling
        bool enable_cuda_profiling = false;
        bool enable_memory_checking = false;
        bool enable_kernel_timing = false;
    };
    
    /**
     * @brief GPU memory usage statistics
     */
    struct GPUMemoryStats {
        size_t total_memory_bytes = 0;
        size_t allocated_memory_bytes = 0;
        size_t free_memory_bytes = 0;
        size_t learning_state_bytes = 0;
        size_t neural_network_bytes = 0;
        size_t temporary_buffer_bytes = 0;
        size_t peak_usage_bytes = 0;
        float fragmentation_ratio = 0.0f;
        float memory_utilization_percent = 0.0f;
    };
    
    /**
     * @brief Performance metrics for CUDA operations
     */
    struct CUDAPerformanceMetrics {
        // Timing metrics (milliseconds)
        float last_update_time_ms = 0.0f;
        float avg_update_time_ms = 0.0f;
        float consolidation_time_ms = 0.0f;
        float state_transfer_time_ms = 0.0f;
        
        // Throughput metrics
        float neurons_per_second = 0.0f;
        float synapses_per_second = 0.0f;
        float memory_bandwidth_gbps = 0.0f;
        
        // Utilization metrics
        float gpu_utilization_percent = 0.0f;
        float memory_utilization_percent = 0.0f;
        float stream_efficiency = 0.0f;
        
        // Error metrics
        uint64_t kernel_launch_errors = 0;
        uint64_t memory_errors = 0;
        uint64_t synchronization_errors = 0;
    };

    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with CUDA configuration
     * @param config CUDA-specific configuration
     */
    explicit NetworkCUDA(const CUDAConfig& config);
    
    /**
     * @brief Destructor with proper GPU cleanup
     */
    virtual ~NetworkCUDA();
    
    /**
     * @brief Initialize CUDA network with neural network configuration
     * @param network_config Neural network configuration
     * @return Success status with error details
     */
    std::pair<bool, std::string> initialize(const NetworkConfig& network_config);
    
    /**
     * @brief Initialize from existing neural network
     * @param network Existing Network instance to copy from
     * @return Success status
     * 
     * Note: This method is deprecated in the modular architecture
     */
    // bool initializeFromNetwork(const Network& network);
    
    /**
     * @brief Reinitialize with new configuration
     * @param network_config New neural network configuration
     * @param preserve_weights Whether to preserve existing weights
     * @return Success status
     */
    bool reinitialize(const NetworkConfig& network_config, bool preserve_weights = false);
    
    /**
     * @brief Check if CUDA is properly initialized
     * @return True if ready for computation
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief Get CUDA device information
     * @return Map of device properties
     */
    std::map<std::string, std::string> getDeviceInfo() const;
    
    // ========================================================================
    // MODULAR ARCHITECTURE INTEGRATION
    // ========================================================================
    
    /**
     * @brief Get number of neurons for architecture sizing
     * @return Total number of neurons
     */
    size_t getNumNeurons() const { return num_neurons_; }
    
    /**
     * @brief Get number of synapses for architecture sizing
     * @return Total number of synapses
     */
    size_t getNumSynapses() const { return num_synapses_; }
    
    /**
     * @brief Copy inputs to GPU (for CorticalModule interface)
     * @param inputs Input vector
     * @return Success status
     */
    bool copyInputsToGPU(const std::vector<float>& inputs);
    
    /**
     * @brief Get neuron outputs (for CorticalModule interface)
     * @return Output vector from neurons
     */
    std::vector<float> getNeuronOutputs() const;
    
    // ========================================================================
    // CORE NEURAL PROCESSING
    // ========================================================================
    
    /**
     * @brief Update neural network on GPU
     * @param dt Time step in seconds
     * @param reward_signal Global reward signal
     * @param novelty_signal Novelty/surprise signal
     */
    void update(float dt, float reward_signal = 0.0f, float novelty_signal = 0.0f);
    
    /**
     * @brief Process input through the network
     * @param inputs Input vector
     * @return Output vector
     */
    std::vector<float> processInput(const std::vector<float>& inputs);
    
    /**
     * @brief Process input with learning
     * @param inputs Input vector
     * @param target_outputs Target outputs for supervised learning
     * @param reward_signal Reward signal for reinforcement learning
     * @return Network outputs
     */
    std::vector<float> processInputWithLearning(const std::vector<float>& inputs,
                                               const std::vector<float>& target_outputs = {},
                                               float reward_signal = 0.0f);
    
    /**
     * @brief Get network outputs
     * @return Current output vector
     */
    std::vector<float> getOutputs() const;
    
    /**
     * @brief Get neuron states
     * @return Vector of neuron membrane potentials
     */
    std::vector<GPUNeuronState> getNeuronStates() const;

    /**
     * @brief Get full synapse state buffer
     * @return Vector of GPUSynapse structures
     */
    std::vector<GPUSynapse> getSynapseStates() const;

    /**
     * @brief Overwrite neuron states on the GPU
     * @param neurons Host-side neuron buffer
     * @return Success status
     */
    bool setNeuronStates(const std::vector<GPUNeuronState>& neurons);

    /**
     * @brief Overwrite synapse states on the GPU
     * @param synapses Host-side synapse buffer
     * @return Success status
     */
    bool setSynapseStates(const std::vector<GPUSynapse>& synapses);

    /**
     * @brief Get device pointer to neurons (for direct GPU access)
     * @return Pointer to GPU neuron array
     */
    GPUNeuronState* getDeviceNeurons() { return d_neurons_; }

    /**
     * @brief Get device pointer to synapses (for direct GPU access)
     * @return Pointer to GPU synapse array
     */
    GPUSynapse* getDeviceSynapses() { return d_synapses_; }

    /**
     * @brief Get synaptic weights
     * @return Vector of all synaptic weights
     */
    std::vector<float> getSynapticWeights() const;
    
    /**
     * @brief Set synaptic weights
     * @param weights New synaptic weights
     * @return Success status
     */
    bool setSynapticWeights(const std::vector<float>& weights);
    
    /**
     * @brief Reset network to initial state
     * @param preserve_topology Whether to keep network structure
     */
    void reset(bool preserve_topology = true);
    
    // ========================================================================
    // LEARNING STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Initialize learning state GPU memory
     * @return Success status with error details
     */
    std::pair<bool, std::string> initializeLearningStateGPU();
    
    /**
     * @brief Update learning state on GPU
     * @param reward_signal Global reward signal
     * @param novelty_signal Novelty/surprise signal
     * @param dt Time step
     */
    void updateLearningStateGPU(float reward_signal, float novelty_signal, float dt);
    
    /**
     * @brief Save learning state from GPU to host buffer
     * @param output_buffer Host buffer for learning state
     * @param buffer_size Size of output buffer
     * @return Success status
     */
    bool saveLearningStateFromGPU(uint8_t* output_buffer, size_t buffer_size);
    
    /**
     * @brief Load learning state from host buffer to GPU
     * @param input_buffer Host buffer containing learning state
     * @param buffer_size Size of input buffer
     * @return Success status
     */
    bool loadLearningStateToGPU(const uint8_t* input_buffer, size_t buffer_size);
    
    /**
     * @brief Save module-specific learning state
     * @param module_id Module identifier
     * @param output_buffer Host buffer for module state
     * @param buffer_size Size of output buffer
     * @return Success status
     */
    bool saveModuleLearningState(int module_id, float* output_buffer, size_t buffer_size);
    
    /**
     * @brief Load module-specific learning state
     * @param module_id Module identifier
     * @param input_buffer Host buffer containing module state
     * @param buffer_size Size of input buffer
     * @return Success status
     */
    bool loadModuleLearningState(int module_id, const float* input_buffer, size_t buffer_size);
    
    /**
     * @brief Calculate learning state buffer size requirements
     * @return Required buffer size in bytes
     */
    size_t calculateLearningStateBufferSize() const;
    
    /**
     * @brief Cleanup learning state GPU memory
     */
    void cleanupLearningStateGPU();
    
    // ========================================================================
    // MEMORY CONSOLIDATION
    // ========================================================================
    
    /**
     * @brief Perform memory consolidation on GPU
     * @param consolidation_strength Consolidation strength (0-1)
     * @return Number of synapses consolidated
     */
    size_t performMemoryConsolidationGPU(float consolidation_strength);
    
    /**
     * @brief Consolidate specific module on GPU
     * @param module_id Module to consolidate
     * @param consolidation_strength Consolidation strength
     * @return Number of synapses consolidated
     */
    size_t consolidateModuleGPU(int module_id, float consolidation_strength);
    
    /**
     * @brief Check if consolidation is needed
     * @return True if consolidation should be performed
     */
    bool shouldConsolidateGPU() const;
    
    /**
     * @brief Get consolidation statistics
     * @return Map of consolidation metrics
     */
    std::map<std::string, float> getConsolidationStats() const;
    
    // ========================================================================
    // INTER-MODULE LEARNING
    // ========================================================================
    
    /**
     * @brief Update inter-module connections on GPU
     * @param learning_rate_multiplier Global learning rate modifier
     * @param dt Time step
     */
    void updateInterModuleConnectionsGPU(float learning_rate_multiplier, float dt);
    
    /**
     * @brief Apply Hebbian learning to inter-module connections
     * @param source_activities Source module activity levels
     * @param target_activities Target module activity levels
     * @param learning_rate Learning rate for connections
     */
    void applyHebbianLearningGPU(const float* source_activities,
                                const float* target_activities,
                                float learning_rate);
    
    /**
     * @brief Get inter-module connection strengths
     * @return Vector of connection strengths
     */
    std::vector<float> getInterModuleConnectionStrengths() const;
    
    /**
     * @brief Set inter-module connection strengths
     * @param strengths New connection strengths
     * @return Success status
     */
    bool setInterModuleConnectionStrengths(const std::vector<float>& strengths);
    
    // ========================================================================
    // NEUROMODULATION
    // ========================================================================
    
    /**
     * @brief Apply neuromodulation on GPU
     * @param dopamine_level Dopamine level
     * @param acetylcholine_level Acetylcholine level
     * @param norepinephrine_level Norepinephrine level
     * @param dt Time step
     */
    void applyNeuromodulationGPU(float dopamine_level, float acetylcholine_level,
                                float norepinephrine_level, float dt);
    
    /**
     * @brief Get neuromodulator levels
     * @return Map of neuromodulator names to levels
     */
    std::map<std::string, float> getNeuromodulatorLevels() const;
    
    /**
     * @brief Set neuromodulator levels
     * @param levels Map of neuromodulator names to levels
     */
    void setNeuromodulatorLevels(const std::map<std::string, float>& levels);
    
    // ========================================================================
    // PERFORMANCE MONITORING AND OPTIMIZATION
    // ========================================================================
    
    /**
     * @brief Get GPU memory usage statistics
     * @return Memory usage statistics
     */
    GPUMemoryStats getMemoryStats() const;
    
    /**
     * @brief Get CUDA performance metrics
     * @return Performance metrics
     */
    CUDAPerformanceMetrics getPerformanceMetrics() const;
    
    /**
     * @brief Optimize GPU memory usage
     * @return Amount of memory freed in bytes
     */
    size_t optimizeMemoryUsage();
    
    /**
     * @brief Profile kernel performance
     * @param enable Enable profiling
     */
    void enableProfiling(bool enable);
    
    /**
     * @brief Get kernel timing information
     * @return Map of kernel names to execution times
     */
    std::map<std::string, float> getKernelTimings() const;
    
    /**
     * @brief Synchronize all CUDA streams
     */
    void synchronizeAllStreams();
    
    /**
     * @brief Check for CUDA errors
     * @return Error status with description
     */
    std::pair<bool, std::string> checkCudaErrors() const;
    
    // ========================================================================
    // MULTI-STREAM PROCESSING
    // ========================================================================
    
    /**
     * @brief Process multiple modules in parallel
     * @param module_inputs Map of module IDs to input vectors
     * @return Map of module IDs to output vectors
     */
    std::map<int, std::vector<float>> processModulesParallel(
        const std::map<int, std::vector<float>>& module_inputs);
    
    /**
     * @brief Update modules in parallel streams
     * @param module_ids Vector of module IDs to update
     * @param dt Time step
     * @param reward_signals Per-module reward signals
     */
    void updateModulesParallel(const std::vector<int>& module_ids, float dt,
                              const std::vector<float>& reward_signals);
    
    /**
     * @brief Get stream utilization
     * @return Map of stream IDs to utilization percentages
     */
    std::map<int, float> getStreamUtilization() const;
    
    // ========================================================================
    // ADVANCED FEATURES
    // ========================================================================
    
    /**
     * @brief Enable mixed precision training
     * @param enable Enable mixed precision
     * @return Success status
     */
    bool enableMixedPrecision(bool enable);
    
    /**
     * @brief Capture CUDA graph for optimization
     * @return Success status
     */
    bool captureComputationGraph();
    
    /**
     * @brief Use captured graph for execution
     * @param use_graph Use captured graph if available
     */
    void useComputationGraph(bool use_graph);
    
    /**
     * @brief Enable tensor core acceleration
     * @param enable Enable tensor cores
     * @return Success status
     */
    bool enableTensorCores(bool enable);
    
    /**
     * @brief Set memory pool size
     * @param size_mb Memory pool size in megabytes
     * @return Success status
     */
    bool setMemoryPoolSize(size_t size_mb);
    
    /**
     * @brief Warm up GPU for optimal performance
     */
    void warmupGPU();
    
    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save complete GPU state to file
     * @param filename Output filename
     * @return Success status
     */
    bool saveGPUState(const std::string& filename) const;
    
    /**
     * @brief Load complete GPU state from file
     * @param filename Input filename
     * @return Success status
     */
    bool loadGPUState(const std::string& filename);
    
    /**
     * @brief Get state size for serialization
     * @return Size in bytes
     */
    size_t getStateSize() const;
    
    /**
     * @brief Validate GPU state integrity
     * @return True if state is valid
     */
    bool validateGPUState() const;
    
    /**
     * @brief Calculate reward prediction error and global stats on CPU
     * @param actual_reward Actual reward received
     * @return Reward prediction error
     */
    float calculateRewardPredictionErrorCPU(float actual_reward);

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    CUDAConfig cuda_config_;
    NetworkConfig network_config_;
    bool is_initialized_ = false;
    
    // CUDA device management
    int device_id_;
    cudaDeviceProp device_properties_;
    cudaStream_t default_stream_ = nullptr;
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> memory_streams_;
    
    // cuBLAS and cuRAND handles
    cublasHandle_t cublas_handle_ = nullptr;
    curandGenerator_t curand_generator_ = nullptr;
    
    // Neural network GPU memory (legacy AoS layout - for persistence compatibility)
    GPUNeuronState* d_neurons_ = nullptr;
    GPUSynapse* d_synapses_ = nullptr;
    float* d_inputs_ = nullptr;
    float* d_outputs_ = nullptr;
    int* d_output_counts_ = nullptr;
    
    // SoA layout (optimized for computation)
    NeuronArrays* d_neuron_arrays_ = nullptr;
    SynapseArrays* d_synapse_arrays_ = nullptr;
    
    // Host copies of SoA structures (containing device pointers) to avoid D2H copies
    NeuronArrays h_neuron_arrays_struct_;
    SynapseArrays h_synapse_arrays_struct_;
    
    bool use_soa_layout_ = true;  // Use SoA by default for performance
    
    // Learning state GPU memory
    GPULearningState* d_learning_state_ = nullptr;
    GPUInterModuleState* d_inter_module_state_ = nullptr;
    
    // Working buffers
    float* d_temp_buffer_ = nullptr;
    float* d_reduction_buffer_ = nullptr;
    int* d_consolidated_count_ = nullptr;
    
    // Host-side buffers for synchronization
    std::vector<float> h_neuron_outputs_;
    std::vector<float> h_synaptic_weights_;
    std::unique_ptr<uint8_t[]> h_learning_state_buffer_;
    size_t learning_state_buffer_size_ = 0;
    
    // Network dimensions
    size_t num_neurons_ = 0;
    size_t num_synapses_ = 0;
    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    size_t num_modules_ = 0;
    int output_group_size_ = 1;
    
    // External references (removed - NetworkCUDA operates independently in modular architecture)
    // Each CorticalModule contains its own NetworkCUDA instance
    
    // Performance monitoring
    mutable CUDAPerformanceMetrics performance_metrics_;
    mutable GPUMemoryStats memory_stats_;
    std::map<std::string, float> kernel_timings_;
    bool profiling_enabled_ = false;
    
    // Graph capture for optimization
    cudaGraph_t computation_graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool use_captured_graph_ = false;
    
    // Memory management
    cudaMemPool_t memory_pool_ = nullptr;
    bool memory_pool_enabled_ = false;
    
    // Thread safety
    mutable std::recursive_mutex cuda_mutex_;
    mutable std::mutex memory_mutex_;
    mutable std::mutex stream_mutex_;
    
    // Error tracking
    mutable std::atomic<uint64_t> total_cuda_errors_{0};
    mutable std::string last_cuda_error_;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_update_time_;
    std::vector<float> update_time_history_;
    float current_time_ = 0.0f;  // Simulation time in milliseconds
    
    // Reward prediction for CPU-based RPE calculation
    float expected_reward_ = 0.0f;
    float reward_history_sum_ = 0.0f;
    int reward_history_count_ = 0;
    static constexpr int MAX_REWARD_HISTORY = 1000;
    
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA device and context
     * @return Success status
     */
    bool initializeCudaDevice();
    
    /**
     * @brief Initialize CUDA streams
     * @return Success status
     */
    bool initializeCudaStreams();
    
    /**
     * @brief Initialize cuBLAS and cuRAND
     * @return Success status
     */
    bool initializeCudaLibraries();
    
    /**
     * @brief Allocate GPU memory for neural network
     * @return Success status
     */
    bool allocateNeuralNetworkMemory();
    
    /**
     * @brief Allocate GPU memory for learning state
     * @return Success status
     */
    bool allocateLearningStateMemory();
    
    /**
     * @brief Allocate working buffers
     * @return Success status
     */
    bool allocateWorkingBuffers();
    
    /**
     * @brief Allocate SoA arrays for neurons
     * @return Success status
     */
    bool allocateNeuronArrays();
    
    /**
     * @brief Allocate SoA arrays for synapses
     * @return Success status
     */
    bool allocateSynapseArrays();
    
    /**
     * @brief Free SoA arrays
     */
    void freeSOAArrays();
    
    /**
     * @brief Convert AoS to SoA layout
     */
    void convertAoSToSoA();
    
    /**
     * @brief Convert SoA to AoS layout (for persistence)
     */
    void convertSoAToAoS();
    
    /**
     * @brief Initialize neural network data on GPU
     * @return Success status
     */
    bool initializeNeuralNetworkData();
    
    /**
     * @brief Initialize learning state data on GPU
     * @return Success status
     */
    bool initializeLearningStateData();
    
    /**
     * @brief Update memory statistics
     */
    void updateMemoryStats() const;
    
    /**
     * @brief Update performance metrics
     * @param kernel_time_ms Time taken for last kernel execution
     */
    void updatePerformanceMetrics(float kernel_time_ms) const;
    
    /**
     * @brief Check and handle CUDA errors
     * @param operation Description of the operation
     * @return True if no errors
     */
    bool checkCudaError(const std::string& operation) const;
    
    /**
     * @brief Cleanup all GPU resources
     */
    void cleanupGPUResources();
    
    /**
     * @brief Get optimal block size for kernel launch
     * @param kernel_func CUDA kernel function pointer
     * @return Optimal block size
     */
    int getOptimalBlockSize(const void* kernel_func) const;
    
    /**
     * @brief Calculate grid size for given problem size
     * @param problem_size Total number of elements to process
     * @param block_size Block size to use
     * @return Grid size
     */
    int calculateGridSize(int problem_size, int block_size) const;
    
    /**
     * @brief Check if synchronization with architecture is needed
     * @return True if synchronization is needed
     */
    bool shouldSynchronize() const;
    
    /**
     * @brief Calculate memory fragmentation ratio
     * @return Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
     */
    float calculateFragmentationRatio() const;
    
    /**
     * @brief Time CUDA kernel execution
     * @param stream CUDA stream to time
     * @return Execution time in milliseconds
     */
    float timeKernelExecution(cudaStream_t stream) const;
    
    /**
     * @brief Validate GPU memory allocation
     * @param ptr GPU memory pointer
     * @param size Expected size
     * @param name Memory region name
     * @return True if valid
     */
    bool validateGPUMemory(void* ptr, size_t size, const std::string& name) const;
    
    /**
     * @brief Synchronize specific stream with timeout
     * @param stream CUDA stream
     * @param timeout_ms Timeout in milliseconds
     * @return True if synchronized successfully
     */
    bool synchronizeStreamWithTimeout(cudaStream_t stream, int timeout_ms = 1000) const;
};

// ============================================================================
// CUDA UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Check if CUDA is available
 * @return True if CUDA is available
 */
bool isCudaAvailable();

/**
 * @brief Get CUDA device count
 * @return Number of CUDA devices
 */
int getCudaDeviceCount();

/**
 * @brief Get optimal CUDA device for neural computation
 * @return Device ID of optimal device
 */
int getOptimalCudaDevice();

/**
 * @brief Get CUDA memory info for device
 * @param device_id Device ID
 * @return Pair of (free_memory, total_memory) in bytes
 */
std::pair<size_t, size_t> getCudaMemoryInfo(int device_id = -1);

/**
 * @brief Warm up CUDA device
 * @param device_id Device ID
 */
void warmUpCudaDevice(int device_id = -1);

/**
 * @brief Get CUDA runtime version
 * @return CUDA runtime version string
 */
std::string getCudaRuntimeVersion();

/**
 * @brief Get CUDA driver version
 * @return CUDA driver version string
 */
std::string getCudaDriverVersion();

#endif // NETWORK_CUDA_H