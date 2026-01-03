#include <engine/NetworkCUDA.cuh>
#include <engine/SplitBrainPlasticityKernels.cuh>
#include <modules/BrainOrchestrator.h>
#include <modules/InterModuleConnection.h>
#include <engine/LearningState.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return retval; \
        } \
    } while(0)

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

NetworkCUDA::NetworkCUDA(const CUDAConfig& config) 
    : cuda_config_(config), device_id_(config.device_id) {
    
    last_update_time_ = std::chrono::high_resolution_clock::now();
    update_time_history_.reserve(1000);
    
    std::cout << "🚀 NetworkCUDA created with device " << device_id_ << std::endl;
}

NetworkCUDA::~NetworkCUDA() {
    cleanupGPUResources();
    std::cout << "🧹 NetworkCUDA cleanup completed" << std::endl;
}

std::pair<bool, std::string> NetworkCUDA::initialize(const NetworkConfig& network_config) {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    try {
        network_config_ = network_config;
        
        // Initialize CUDA device and context
        if (!initializeCudaDevice()) {
            return {false, "Failed to initialize CUDA device"};
        }
        
        // Initialize CUDA streams
        if (!initializeCudaStreams()) {
            return {false, "Failed to initialize CUDA streams"};
        }
        
        // Initialize cuBLAS and cuRAND
        if (!initializeCudaLibraries()) {
            return {false, "Failed to initialize CUDA libraries"};
        }
        
        // Set network dimensions from config
        num_neurons_ = network_config_.num_neurons;
        if (network_config_.num_inputs > 0) {
            num_inputs_ = network_config_.num_inputs;
        } else {
            num_inputs_ = std::max<size_t>(1, num_neurons_ / 4);
        }

        if (network_config_.num_outputs > 0) {
            num_outputs_ = network_config_.num_outputs;
        } else {
            num_outputs_ = std::max<size_t>(1, num_neurons_ / 4);
        }

        if (network_config_.num_synapses > 0) {
            num_synapses_ = network_config_.num_synapses;
        } else {
            size_t fanout = static_cast<size_t>(
                std::max<float>(1.0f, network_config_.connection_density * static_cast<float>(num_neurons_)));
            fanout = std::min<size_t>(fanout, num_neurons_);
            num_synapses_ = std::max<size_t>(fanout * num_neurons_, num_neurons_);
        }

        output_group_size_ = std::max<int>(1,
            static_cast<int>((num_neurons_ + num_outputs_ - 1) / num_outputs_));
        
        // Allocate GPU memory
        if (!allocateNeuralNetworkMemory()) {
            return {false, "Failed to allocate neural network GPU memory"};
        }
        
        if (!allocateWorkingBuffers()) {
            return {false, "Failed to allocate working buffers"};
        }
        
        // Initialize neural network data
        if (!initializeNeuralNetworkData()) {
            return {false, "Failed to initialize neural network data"};
        }
        
        // Initialize learning state if enabled
        if (cuda_config_.enable_learning_state_gpu) {
            auto [success, error_msg] = initializeLearningStateGPU();
            if (!success) {
                return {false, "Failed to initialize learning state GPU: " + error_msg};
            }
        }
        
        // Initialize host buffers
        h_neuron_outputs_.resize(num_outputs_);
        h_synaptic_weights_.resize(num_synapses_);
        
        // Warm up GPU
        warmupGPU();
        
        is_initialized_ = true;
        
        std::cout << "✅ NetworkCUDA initialized successfully" << std::endl;
        std::cout << "   Neurons: " << num_neurons_ << ", Synapses: " << num_synapses_ << std::endl;
        std::cout << "   GPU Memory: " << getMemoryStats().allocated_memory_bytes / (1024*1024) << " MB" << std::endl;
        
        return {true, "Success"};
        
    } catch (const std::exception& e) {
        return {false, "Exception during initialization: " + std::string(e.what())};
    }
}

bool NetworkCUDA::initializeCudaDevice() {
    // Set flag to yield CPU while waiting for GPU to complete tasks.
    // This prevents 100% CPU usage during cudaDeviceSynchronize().
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Get device properties
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device_id_));
    
    std::cout << "🔧 CUDA Device: " << device_properties_.name << std::endl;
    std::cout << "   Compute Capability: " << device_properties_.major << "." << device_properties_.minor << std::endl;
    std::cout << "   Global Memory: " << device_properties_.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "   Shared Memory per Block: " << device_properties_.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "   Max Threads per Block: " << device_properties_.maxThreadsPerBlock << std::endl;
    
    // Enable unified memory if requested
    if (cuda_config_.enable_unified_memory && device_properties_.unifiedAddressing) {
        std::cout << "🔗 Unified memory enabled" << std::endl;
    }
    
    // Create memory pool if enabled
    if (cuda_config_.enable_memory_pool && device_properties_.major >= 6) {
        cudaMemPoolProps pool_props = {};
        pool_props.allocType = cudaMemAllocationTypePinned;
        pool_props.handleTypes = cudaMemHandleTypeNone;
        pool_props.location.type = cudaMemLocationTypeDevice;
        pool_props.location.id = device_id_;
        
        if (cudaMemPoolCreate(&memory_pool_, &pool_props) == cudaSuccess) {
            size_t pool_size = cuda_config_.memory_pool_size_mb * 1024 * 1024;
            cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrReservedMemHigh, &pool_size);
            memory_pool_enabled_ = true;
            std::cout << "💾 Memory pool created: " << cuda_config_.memory_pool_size_mb << " MB" << std::endl;
        }
    }
    
    return true;
}

bool NetworkCUDA::initializeCudaStreams() {
    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&default_stream_));
    
    // Create compute streams
    compute_streams_.resize(cuda_config_.num_compute_streams);
    for (int i = 0; i < cuda_config_.num_compute_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&compute_streams_[i]));
        
        // Set stream priority if enabled
        if (cuda_config_.enable_stream_priorities) {
            int priority = (i == 0) ? -1 : 0; // High priority for main compute stream
            cudaStreamCreateWithPriority(&compute_streams_[i], cudaStreamDefault, priority);
        }
    }
    
    // Create memory streams
    memory_streams_.resize(cuda_config_.num_memory_streams);
    for (int i = 0; i < cuda_config_.num_memory_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&memory_streams_[i]));
    }
    
    std::cout << "🌊 Created " << compute_streams_.size() << " compute streams and " 
              << memory_streams_.size() << " memory streams" << std::endl;
    
    return true;
}

bool NetworkCUDA::initializeCudaLibraries() {
    // Initialize cuBLAS
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "❌ Failed to create cuBLAS handle" << std::endl;
        return false;
    }
    
    // Set cuBLAS stream
    cublasSetStream(cublas_handle_, default_stream_);
    
    // Enable tensor cores if available and requested
    if (cuda_config_.enable_tensor_cores && device_properties_.major >= 7) {
        cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
        std::cout << "⚡ Tensor cores enabled" << std::endl;
    }
    
    // Initialize cuRAND
    if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::cerr << "❌ Failed to create cuRAND generator" << std::endl;
        return false;
    }
    
    curandSetStream(curand_generator_, default_stream_);
    curandSetPseudoRandomGeneratorSeed(curand_generator_, time(nullptr));
    
    std::cout << "📚 CUDA libraries initialized" << std::endl;
    return true;
}

// ============================================================================
// MEMORY ALLOCATION AND MANAGEMENT
// ============================================================================

bool NetworkCUDA::allocateNeuralNetworkMemory() {
    try {
        // Allocate neuron states
        size_t neuron_size = num_neurons_ * sizeof(GPUNeuronState);
        std::cout << "   Requesting neuron buffer: " << neuron_size / (1024 * 1024) << " MB ("
                  << num_neurons_ << " neurons)" << std::endl;
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaMallocFromPoolAsync(&d_neurons_, neuron_size, memory_pool_, default_stream_));
        } else {
            cudaError_t status = cudaMalloc(&d_neurons_, neuron_size);
            if (status != cudaSuccess) {
                std::cerr << "❌ cudaMalloc neurons failed (" << neuron_size / (1024.0 * 1024.0)
                          << " MB): " << cudaGetErrorString(status) << std::endl;
                return false;
            }
        }

        // Allocate synapses
        size_t synapse_size = num_synapses_ * sizeof(GPUSynapse);
        std::cout << "   Requesting synapse buffer: " << synapse_size / (1024 * 1024) << " MB ("
                  << num_synapses_ << " synapses)" << std::endl;
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaMallocFromPoolAsync(&d_synapses_, synapse_size, memory_pool_, default_stream_));
        } else {
            cudaError_t status = cudaMalloc(&d_synapses_, synapse_size);
            if (status != cudaSuccess) {
                std::cerr << "❌ cudaMalloc synapses failed (" << synapse_size / (1024.0 * 1024.0)
                          << " MB): " << cudaGetErrorString(status) << std::endl;
                return false;
            }
        }

        // CRITICAL: Synchronize stream after async allocations
        // This ensures memory allocations complete before being used
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaStreamSynchronize(default_stream_));
        }

        // Allocate input/output buffers
        CUDA_CHECK(cudaMalloc(&d_inputs_, num_inputs_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs_, num_outputs_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_counts_, num_outputs_ * sizeof(int)));

        std::cout << "💾 Allocated neural network GPU memory:" << std::endl;
        std::cout << "   Neurons: " << neuron_size / (1024*1024) << " MB" << std::endl;
        std::cout << "   Synapses: " << synapse_size / (1024*1024) << " MB" << std::endl;
        
        // Allocate SoA arrays if enabled
        if (use_soa_layout_) {
            if (!allocateNeuronArrays()) {
                std::cerr << "❌ Failed to allocate neuron SoA arrays" << std::endl;
                return false;
            }
            if (!allocateSynapseArrays()) {
                std::cerr << "❌ Failed to allocate synapse SoA arrays" << std::endl;
                return false;
            }
            std::cout << "⚡ SoA layout allocated for optimized GPU coalescing" << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating neural network memory: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::allocateNeuronArrays() {
    try {
        // Initialize host-side structure
        h_neuron_arrays_struct_.num_neurons = num_neurons_;
        
        // Allocate device-side structure
        CUDA_CHECK(cudaMalloc(&d_neuron_arrays_, sizeof(NeuronArrays)));
        
        // Allocate core membrane dynamics (hot path - most frequently accessed)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.V, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.u, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.I_syn_0, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.I_syn_1, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.I_syn_2, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.I_syn_3, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.I_ext, num_neurons_ * sizeof(float)));
        
        // Allocate calcium dynamics (hot path)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.ca_conc_0, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.ca_conc_1, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.ca_conc_2, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.ca_conc_3, num_neurons_ * sizeof(float)));
        
        // Allocate timing (hot path)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.last_spike_time, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.previous_spike_time, num_neurons_ * sizeof(float)));
        
        // Allocate activity (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.average_firing_rate, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.average_activity, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.activity_level, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.firing_rate, num_neurons_ * sizeof(float)));
        
        // Allocate plasticity (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.excitability, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.synaptic_scaling_factor, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.bcm_threshold, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.plasticity_threshold, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.threshold, num_neurons_ * sizeof(float)));
        
        // Allocate neuromodulation (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.dopamine_concentration, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.acetylcholine_level, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.serotonin_level, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.norepinephrine_level, num_neurons_ * sizeof(float)));
        
        // Allocate ion channels (cold path - less frequently accessed)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.na_m, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.na_h, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.k_n, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.ca_channel_state, num_neurons_ * sizeof(float)));
        
        // Allocate network properties (cold path)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.neuron_type, num_neurons_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.layer_id, num_neurons_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.column_id, num_neurons_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.active, num_neurons_ * sizeof(int)));
        
        // Allocate metabolism (cold path)
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.energy_level, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_neuron_arrays_struct_.metabolic_demand, num_neurons_ * sizeof(float)));
        
        // Copy structure to device
        CUDA_CHECK(cudaMemcpy(d_neuron_arrays_, &h_neuron_arrays_struct_, sizeof(NeuronArrays), cudaMemcpyHostToDevice));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating neuron SoA: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::allocateSynapseArrays() {
    try {
        // Initialize host-side structure
        h_synapse_arrays_struct_.num_synapses = num_synapses_;
        
        // Allocate device-side structure
        CUDA_CHECK(cudaMalloc(&d_synapse_arrays_, sizeof(SynapseArrays)));
        
        // Allocate connectivity (hot path)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.pre_neuron_idx, num_synapses_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.post_neuron_idx, num_synapses_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.post_compartment, num_synapses_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.active, num_synapses_ * sizeof(int)));
        
        // Allocate synaptic properties (hot path)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.weight, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.max_weight, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.min_weight, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.effective_weight, num_synapses_ * sizeof(float)));
        
        // Allocate plasticity (hot path)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.eligibility_trace, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.learning_rate, num_synapses_ * sizeof(float)));
        
        // Allocate timing (hot path)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.last_pre_spike_time, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.last_post_spike_time, num_synapses_ * sizeof(float)));
        
        // Allocate neuromodulation (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.dopamine_sensitivity, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.dopamine_level, num_synapses_ * sizeof(float)));
        
        // Allocate short-term plasticity (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.release_probability, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.facilitation_factor, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.depression_factor, num_synapses_ * sizeof(float)));
        
        // Allocate calcium dynamics (medium frequency)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.presynaptic_calcium, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.postsynaptic_calcium, num_synapses_ * sizeof(float)));
        
        // Allocate other properties (cold path)
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.plasticity_modulation, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.metaplasticity_factor, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.delay, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.homeostatic_scaling, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.receptor_index, num_synapses_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_synapse_arrays_struct_.vesicle_count, num_synapses_ * sizeof(int)));
        
        // Copy structure to device
        CUDA_CHECK(cudaMemcpy(d_synapse_arrays_, &h_synapse_arrays_struct_, sizeof(SynapseArrays), cudaMemcpyHostToDevice));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating synapse SoA: " << e.what() << std::endl;
        return false;
    }
}

std::pair<bool, std::string> NetworkCUDA::initializeLearningStateGPU() {
    try {
        if (!allocateLearningStateMemory()) {
            return {false, "Failed to allocate learning state memory"};
        }
        
        if (!initializeLearningStateData()) {
            return {false, "Failed to initialize learning state data"};
        }
        
        // Calculate buffer size for host-GPU transfers
        learning_state_buffer_size_ = calculateLearningStateBufferSize();
        h_learning_state_buffer_ = std::make_unique<uint8_t[]>(learning_state_buffer_size_);
        
        std::cout << "🧠 Learning state GPU initialized:" << std::endl;
        std::cout << "   Buffer size: " << learning_state_buffer_size_ / (1024*1024) << " MB" << std::endl;
        
        return {true, "Success"};
        
    } catch (const std::exception& e) {
        return {false, "Exception: " + std::string(e.what())};
    }
}

bool NetworkCUDA::allocateLearningStateMemory() {
    try {
        // Allocate main learning state structure
        CUDA_CHECK(cudaMalloc(&d_learning_state_, sizeof(GPULearningState)));
        
        // Allocate inter-module state structure
        CUDA_CHECK(cudaMalloc(&d_inter_module_state_, sizeof(GPUInterModuleState)));
        
        // Create host-side structures to set up GPU pointers
        GPULearningState h_learning_state;
        GPUInterModuleState h_inter_module_state;
        
        // Allocate learning trace arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.eligibility_traces, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.synaptic_tags, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.consolidation_weights, num_synapses_ * sizeof(float)));
        
        // Allocate neuromodulator arrays (3 per neuron: dopamine, acetylcholine, norepinephrine)
        CUDA_CHECK(cudaMalloc(&h_learning_state.neuromodulator_levels, num_neurons_ * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.firing_rate_history, num_neurons_ * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.prediction_errors, num_neurons_ * 100 * sizeof(float)));
        
        // Allocate learning parameter arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.learning_rates, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.plasticity_thresholds, num_neurons_ * sizeof(float)));
        
        // Allocate performance tracking arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.learning_step_counts, num_neurons_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.reward_history, num_neurons_ * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.history_indices, num_neurons_ * sizeof(uint32_t)));
        
        // Allocate module assignment arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.module_assignments, num_neurons_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.module_boundaries, 32 * sizeof(int))); // Max 16 modules
        h_learning_state.num_modules = 0; // Will be set by brain architecture
        
        // Initialize arrays to zero
        CUDA_CHECK(cudaMemset(h_learning_state.eligibility_traces, 0, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_learning_state.synaptic_tags, 0, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_learning_state.neuromodulator_levels, 0, num_neurons_ * 3 * sizeof(float)));
        
        // Initialize learning parameters to default values
        std::vector<float> default_learning_rates(num_neurons_, 0.001f);
        CUDA_CHECK(cudaMemcpy(h_learning_state.learning_rates, default_learning_rates.data(), 
                             num_neurons_ * sizeof(float), cudaMemcpyHostToDevice));
        
        std::vector<float> default_thresholds(num_neurons_, 0.1f);
        CUDA_CHECK(cudaMemcpy(h_learning_state.plasticity_thresholds, default_thresholds.data(),
                             num_neurons_ * sizeof(float), cudaMemcpyHostToDevice));
        
        // Copy structure to GPU
        CUDA_CHECK(cudaMemcpy(d_learning_state_, &h_learning_state, sizeof(GPULearningState), cudaMemcpyHostToDevice));
        
        // Initialize inter-module state (will be configured by brain architecture)
        size_t max_connections = 100; // Initial allocation, can be expanded
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.connection_strengths, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.usage_frequencies, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.correlation_strengths, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.activation_counts, max_connections * sizeof(uint64_t)));
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.pre_synaptic_traces, max_connections * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.post_synaptic_traces, max_connections * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.timing_differences, max_connections * 100 * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.source_modules, max_connections * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.target_modules, max_connections * sizeof(int)));
        h_inter_module_state.num_connections = 0; // Will be set by brain architecture
        
        // Initialize connection strengths to small values
        std::vector<float> initial_strengths(max_connections, 0.1f);
        CUDA_CHECK(cudaMemcpy(h_inter_module_state.connection_strengths, initial_strengths.data(),
                             max_connections * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_inter_module_state_, &h_inter_module_state, sizeof(GPUInterModuleState), cudaMemcpyHostToDevice));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating learning state memory: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// CORE NEURAL PROCESSING
// ============================================================================

void NetworkCUDA::update(float dt, float reward_signal, float novelty_signal) {
    if (!is_initialized_) {
        std::cerr << "❌ NetworkCUDA not initialized" << std::endl;
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Calculate reward prediction error on CPU (avoids single-threaded GPU bottleneck)
        float rpe = calculateRewardPredictionErrorCPU(reward_signal);
        
        // Use optimized fused kernels if SoA layout is enabled
        if (use_soa_layout_ && d_neuron_arrays_ && d_synapse_arrays_) {
            // FUSED NEURON UPDATE: combines neuron dynamics + calcium + neuromodulation
            float dopamine = 0.5f;  // Default dopamine level (could be dynamic)
            float serotonin = 0.5f; // Default serotonin level
            KernelLaunchWrappers::launch_fused_neuron_update(
                d_neuron_arrays_, current_time_, dt, dopamine, serotonin, num_neurons_
            );
            
            // FUSED PLASTICITY: combines STDP + eligibility + reward modulation
            // Use RPE instead of raw reward for better learning
            KernelLaunchWrappers::launch_fused_plasticity(
                d_synapse_arrays_, d_neuron_arrays_, rpe, current_time_, dt, num_synapses_
            );
        } else {
            // Fallback to separate kernel launches (legacy AoS path)
            // Update neuron states
            KernelLaunchWrappers::update_neuron_states(d_neurons_, current_time_, dt, num_neurons_);

            // Update calcium dynamics
            KernelLaunchWrappers::update_calcium_dynamics(d_neurons_, current_time_, dt, num_neurons_);

            // Run STDP and update eligibility traces
            KernelLaunchWrappers::run_stdp_and_eligibility(d_synapses_, d_neurons_, current_time_, dt, num_synapses_);

            // Apply reward and adaptation if there's a reward signal
            // Use RPE instead of raw reward for better learning
            if (std::abs(rpe) > 0.001f) {
                KernelLaunchWrappers::apply_reward_and_adaptation(d_synapses_, d_neurons_, rpe, current_time_, dt, num_synapses_);
            }
        }

        // Update learning state if enabled
        if (cuda_config_.enable_learning_state_gpu && d_learning_state_) {
            updateLearningStateGPU(reward_signal, novelty_signal, dt);
        }

        // Run homeostatic mechanisms (still separate for now)
        if (!use_soa_layout_) {
            KernelLaunchWrappers::run_homeostatic_mechanisms(d_neurons_, d_synapses_, current_time_, num_neurons_, num_synapses_);
        }

        // Compute lightweight neuron outputs on-device
        if (d_output_counts_ && d_outputs_) {
            KernelLaunchWrappers::compute_neuron_outputs(
                d_neurons_,
                d_outputs_,
                d_output_counts_,
                static_cast<int>(num_neurons_),
                static_cast<int>(num_outputs_),
                output_group_size_);
        }

        // Synchronize default stream
        CUDA_CHECK_RETURN(cudaStreamSynchronize(default_stream_), void());

        // Advance simulation time
        current_time_ += dt;

        // Update performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        float update_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        updatePerformanceMetrics(update_time_ms);

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during GPU update: " << e.what() << std::endl;
    }
}

void NetworkCUDA::applyPlasticity(float dt, float reward_signal, 
                                  const neurogen::LearningModeParams& learning_params) {
    if (!is_initialized_ || !d_synapses_) {
        return;
    }
    
    // Prepare kernel parameters
    neurogen::kernels::SplitBrainPlasticityParams params;
    
    // Convert learning mode to device enum
    switch (learning_params.mode) {
        case neurogen::LearningMode::PURE_STDP:
            params.mode = neurogen::kernels::DeviceLearningMode::PURE_STDP;
            break;
        case neurogen::LearningMode::REWARD_MODULATED_STDP:
            params.mode = neurogen::kernels::DeviceLearningMode::REWARD_MODULATED_STDP;
            break;
        case neurogen::LearningMode::MIXED_STDP:
            params.mode = neurogen::kernels::DeviceLearningMode::MIXED_STDP;
            break;
    }
    
    params.stdp_learning_rate = learning_params.stdp_learning_rate;
    params.stdp_tau_positive = learning_params.stdp_tau_positive;
    params.stdp_tau_negative = learning_params.stdp_tau_negative;
    params.stdp_a_positive = learning_params.stdp_a_positive;
    params.stdp_a_negative = learning_params.stdp_a_negative;
    params.reward_sensitivity = learning_params.reward_sensitivity;
    params.eligibility_decay = learning_params.eligibility_decay;
    params.unsupervised_weight = learning_params.unsupervised_weight;
    params.supervised_weight = learning_params.supervised_weight;
    
    // Launch appropriate kernel based on mode
    int threads = 256;
    int blocks = (num_synapses_ + threads - 1) / threads;
    
    if (params.mode == neurogen::kernels::DeviceLearningMode::PURE_STDP) {
        // Pure STDP: Completely ignore reward
        neurogen::kernels::pureSTDPKernel<<<blocks, threads>>>(
            d_synapses_,
            d_neurons_,
            params,
            current_time_,
            dt,
            num_synapses_
        );
    } else if (params.mode == neurogen::kernels::DeviceLearningMode::REWARD_MODULATED_STDP) {
        // Reward-modulated: Two-stage process
        // 1. Update eligibility traces
        neurogen::kernels::updateEligibilityTracesKernel<<<blocks, threads>>>(
            d_synapses_,
            d_neurons_,
            d_eligibility_traces_,
            params,
            current_time_,
            dt,
            num_synapses_
        );
        
        // 2. Apply reward-gated learning
        float rpe = calculateRewardPredictionError(reward_signal);
        neurogen::kernels::rewardModulatedPlasticityKernel<<<blocks, threads>>>(
            d_synapses_,
            d_eligibility_traces_,
            params,
            rpe,
            dt,
            num_synapses_
        );
    } else {
        // Mixed mode: Use unified kernel
        neurogen::kernels::splitBrainPlasticityKernel<<<blocks, threads>>>(
            d_synapses_,
            d_neurons_,
            d_eligibility_traces_,
            params,
            reward_signal,
            current_time_,
            dt,
            num_synapses_
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> NetworkCUDA::processInput(const std::vector<float>& inputs) {
    return processInputWithLearning(inputs, {}, 0.0f);
}

std::vector<float> NetworkCUDA::processInputWithLearning(const std::vector<float>& inputs,
                                                        const std::vector<float>& target_outputs,
                                                        float reward_signal) {
    if (!is_initialized_) {
        std::cerr << "❌ NetworkCUDA not initialized" << std::endl;
        return {};
    }
    
    try {
        // Copy inputs to GPU
        size_t input_size = std::min(inputs.size(), num_inputs_);
        if (input_size > 0) {
        std::lock_guard<std::recursive_mutex> input_lock(cuda_mutex_);
            CUDA_CHECK_RETURN(cudaMemcpy(d_inputs_, inputs.data(), input_size * sizeof(float), cudaMemcpyHostToDevice), {});
        }
        
        // Process through network (this would involve more complex processing in practice)
        // For now, we'll simulate by updating neurons and getting outputs
        update(0.001f, reward_signal, 0.0f);
        
        // Copy outputs from GPU
        {
        std::lock_guard<std::recursive_mutex> output_lock(cuda_mutex_);
            CUDA_CHECK_RETURN(cudaMemcpyAsync(
                h_neuron_outputs_.data(),
                d_outputs_,
                num_outputs_ * sizeof(float),
                cudaMemcpyDeviceToHost,
                default_stream_), {});

            CUDA_CHECK_RETURN(cudaStreamSynchronize(default_stream_), {});
        }
        
        return h_neuron_outputs_;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error processing input: " << e.what() << std::endl;
        return {};
    }
}

// ============================================================================
// LEARNING STATE MANAGEMENT
// ============================================================================

void NetworkCUDA::updateLearningStateGPU(float reward_signal, float novelty_signal, float dt) {
    if (!d_learning_state_ || !d_inter_module_state_) {
        return;
    }
    
    try {
        // Update eligibility traces
        LearningStateKernels::update_eligibility_traces(
            d_learning_state_, d_neurons_, d_synapses_, reward_signal, dt, 
            num_neurons_, num_synapses_);
        
        // Apply synaptic tagging
        LearningStateKernels::apply_synaptic_tagging(
            d_learning_state_, d_neurons_, novelty_signal, dt, 
            num_neurons_, num_synapses_);
        
        // Update neuromodulators based on reward and context
        float dopamine = reward_signal; // Reward prediction error
        float acetylcholine = std::abs(novelty_signal); // Attention/uncertainty
        float norepinephrine = std::abs(reward_signal) > 0.5f ? 1.0f : 0.0f; // Arousal
        
        LearningStateKernels::update_neuromodulators(
            d_learning_state_, d_neurons_, dopamine, acetylcholine, norepinephrine, dt, num_neurons_);
        
        // Update inter-module connections if we have module information
        // Note: brain_architecture_ removed in modular architecture
        if (false) {  // Disabled - each module operates independently
            LearningStateKernels::update_inter_module_connections(
                d_inter_module_state_, d_learning_state_, d_neurons_, 1.0f, dt, num_neurons_);
        }
        
        // Update learning statistics
        LearningStateKernels::update_learning_statistics(
            d_learning_state_, d_neurons_, reward_signal, novelty_signal, dt, num_neurons_);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error updating learning state GPU: " << e.what() << std::endl;
    }
}

size_t NetworkCUDA::performMemoryConsolidationGPU(float consolidation_strength) {
    if (!d_learning_state_) {
        return 0;
    }
    
    try {
        // Allocate device memory for consolidation counter
        int* d_consolidated_count;
        CUDA_CHECK_RETURN(cudaMalloc(&d_consolidated_count, sizeof(int)), 0);
        
        // Perform consolidation
        LearningStateKernels::perform_memory_consolidation(
            d_learning_state_, d_neurons_, d_synapses_, consolidation_strength,
            d_consolidated_count, num_neurons_, num_synapses_);
        
        // Copy result back to host
        int h_consolidated_count;
        CUDA_CHECK_RETURN(cudaMemcpy(&h_consolidated_count, d_consolidated_count, sizeof(int), cudaMemcpyDeviceToHost), 0);
        
        CUDA_CHECK_RETURN(cudaFree(d_consolidated_count), 0);
        
        std::cout << "🧠 GPU memory consolidation completed: " << h_consolidated_count << " synapses consolidated" << std::endl;
        
        return static_cast<size_t>(h_consolidated_count);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during memory consolidation: " << e.what() << std::endl;
        return 0;
    }
}

// ============================================================================
// MODULAR ARCHITECTURE HELPER METHODS
// ============================================================================

bool NetworkCUDA::copyInputsToGPU(const std::vector<float>& inputs) {
    if (!is_initialized_ || !d_inputs_) {
        return false;
    }
    
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    size_t input_size = std::min(inputs.size(), num_inputs_);
    if (input_size > 0) {
        cudaError_t error = cudaMemcpy(d_inputs_, inputs.data(), 
                                      input_size * sizeof(float), 
                                      cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy inputs to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    return true;
}

std::vector<float> NetworkCUDA::getNeuronOutputs() const {
    if (!is_initialized_ || !d_outputs_) {
        return {};
    }
    
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    std::vector<float> outputs(num_outputs_);
    cudaError_t error = cudaMemcpy(outputs.data(),
                                   d_outputs_,
                                   num_outputs_ * sizeof(float),
                                   cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        std::cerr << "Failed to get neuron outputs: " << cudaGetErrorString(error) << std::endl;
        return {};
    }
    
    return outputs;
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

NetworkCUDA::GPUMemoryStats NetworkCUDA::getMemoryStats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    GPUMemoryStats stats;
    
    // Get total GPU memory
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        stats.total_memory_bytes = total_bytes;
        stats.free_memory_bytes = free_bytes;
        stats.allocated_memory_bytes = total_bytes - free_bytes;
    }
    
    // Calculate specific allocations
    if (is_initialized_) {
        stats.neural_network_bytes = num_neurons_ * sizeof(GPUNeuronState) + num_synapses_ * sizeof(GPUSynapse);
        stats.neural_network_bytes += (num_inputs_ + num_outputs_) * sizeof(float);
        
        if (d_learning_state_) {
            stats.learning_state_bytes = num_synapses_ * 3 * sizeof(float); // eligibility + tags + consolidation
            stats.learning_state_bytes += num_neurons_ * (3 + 1000 + 100) * sizeof(float); // neuromod + history + errors
            stats.learning_state_bytes += num_neurons_ * (2 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t)); // params + counts
        }
        
        stats.temporary_buffer_bytes = learning_state_buffer_size_;
    }
    
    stats.memory_utilization_percent = (static_cast<float>(stats.allocated_memory_bytes) / stats.total_memory_bytes) * 100.0f;
    stats.fragmentation_ratio = calculateFragmentationRatio();
    
    return stats;
}

NetworkCUDA::CUDAPerformanceMetrics NetworkCUDA::getPerformanceMetrics() const {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    return performance_metrics_;
}

void NetworkCUDA::updatePerformanceMetrics(float kernel_time_ms) const {
    performance_metrics_.last_update_time_ms = kernel_time_ms;
    
    // Update average with exponential moving average
    if (performance_metrics_.avg_update_time_ms == 0.0f) {
        performance_metrics_.avg_update_time_ms = kernel_time_ms;
    } else {
        performance_metrics_.avg_update_time_ms = 0.9f * performance_metrics_.avg_update_time_ms + 0.1f * kernel_time_ms;
    }
    
    // Calculate throughput metrics
    if (kernel_time_ms > 0.0f) {
        performance_metrics_.neurons_per_second = static_cast<float>(num_neurons_) / (kernel_time_ms / 1000.0f);
        performance_metrics_.synapses_per_second = static_cast<float>(num_synapses_) / (kernel_time_ms / 1000.0f);
    }
    
    // Update memory bandwidth (simplified calculation)
    size_t data_transferred = (num_neurons_ * sizeof(GPUNeuronState) + num_synapses_ * sizeof(GPUSynapse));
    if (kernel_time_ms > 0.0f) {
        performance_metrics_.memory_bandwidth_gbps = (static_cast<float>(data_transferred) / (1024*1024*1024)) / (kernel_time_ms / 1000.0f);
    }
    
    // Update utilization metrics
    updateMemoryStats();
    performance_metrics_.memory_utilization_percent = getMemoryStats().memory_utilization_percent;
}

// ============================================================================
// DATA ACCESS METHODS (FIXED FOR LOW VRAM)
// ============================================================================

std::vector<GPUNeuronState> NetworkCUDA::getNeuronStates() const {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    if (!is_initialized_) return {};
    
    std::vector<GPUNeuronState> host_neurons(num_neurons_);
    
    // MEMORY OPTIMIZATION: Assemble on CPU to avoid VRAM OOM
    if (use_soa_layout_ && d_neuron_arrays_) {
        // We assume the SoA data on GPU is the source of truth.
        // We download arrays one by one to minimize peak RAM usage.
        
        NeuronArrays& gpu_ptrs = const_cast<NetworkCUDA*>(this)->h_neuron_arrays_struct_;
        std::vector<float> temp_buffer(num_neurons_);
        std::vector<int> temp_int_buffer(num_neurons_);
        
        // Helper macro for float fields
        #define FETCH_FLOAT(field, gpu_ptr) \
            cudaMemcpy(temp_buffer.data(), gpu_ptr, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost); \
            for(size_t i=0; i<num_neurons_; ++i) host_neurons[i].field = temp_buffer[i];

        // Helper for int fields
        #define FETCH_INT(field, gpu_ptr) \
            cudaMemcpy(temp_int_buffer.data(), gpu_ptr, num_neurons_ * sizeof(int), cudaMemcpyDeviceToHost); \
            for(size_t i=0; i<num_neurons_; ++i) host_neurons[i].field = temp_int_buffer[i];

        // Core Dynamics
        FETCH_FLOAT(V, gpu_ptrs.V);
        FETCH_FLOAT(u, gpu_ptrs.u);
        FETCH_FLOAT(I_ext, gpu_ptrs.I_ext);
        
        // Synaptic Inputs (Array in struct, handle manually)
        cudaMemcpy(temp_buffer.data(), gpu_ptrs.I_syn_0, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
        for(size_t i=0; i<num_neurons_; ++i) host_neurons[i].I_syn[0] = temp_buffer[i];
        
        cudaMemcpy(temp_buffer.data(), gpu_ptrs.I_syn_1, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
        for(size_t i=0; i<num_neurons_; ++i) host_neurons[i].I_syn[1] = temp_buffer[i];
        
        // Calcium
        cudaMemcpy(temp_buffer.data(), gpu_ptrs.ca_conc_0, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
        for(size_t i=0; i<num_neurons_; ++i) host_neurons[i].ca_conc[0] = temp_buffer[i];

        // Timing & Activity
        FETCH_FLOAT(last_spike_time, gpu_ptrs.last_spike_time);
        FETCH_FLOAT(previous_spike_time, gpu_ptrs.previous_spike_time);
        FETCH_FLOAT(average_firing_rate, gpu_ptrs.average_firing_rate);
        FETCH_FLOAT(average_activity, gpu_ptrs.average_activity);
        FETCH_FLOAT(activity_level, gpu_ptrs.activity_level);
        FETCH_FLOAT(firing_rate, gpu_ptrs.firing_rate);
        
        // Plasticity
        FETCH_FLOAT(excitability, gpu_ptrs.excitability);
        FETCH_FLOAT(synaptic_scaling_factor, gpu_ptrs.synaptic_scaling_factor);
        FETCH_FLOAT(threshold, gpu_ptrs.threshold);
        FETCH_FLOAT(bcm_threshold, gpu_ptrs.bcm_threshold);
        FETCH_FLOAT(plasticity_threshold, gpu_ptrs.plasticity_threshold);
        
        // Neuromodulation
        FETCH_FLOAT(dopamine_concentration, gpu_ptrs.dopamine_concentration);
        FETCH_FLOAT(acetylcholine_level, gpu_ptrs.acetylcholine_level);
        FETCH_FLOAT(serotonin_level, gpu_ptrs.serotonin_level);
        FETCH_FLOAT(norepinephrine_level, gpu_ptrs.norepinephrine_level);
        
        // Properties
        FETCH_INT(neuron_type, gpu_ptrs.neuron_type);
        FETCH_INT(active, gpu_ptrs.active);
        
        // Cleanup macros
        #undef FETCH_FLOAT
        #undef FETCH_INT
        
    } else if (d_neurons_) {
        // Fallback for legacy AoS mode
        cudaMemcpy(host_neurons.data(), d_neurons_, num_neurons_ * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    }
    
    return host_neurons;
}

std::vector<GPUSynapse> NetworkCUDA::getSynapseStates() const {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);

    if (!is_initialized_) return {};

    std::vector<GPUSynapse> host_synapses(num_synapses_);

    // MEMORY OPTIMIZATION: Assemble on CPU to avoid VRAM OOM
    // This is critical for 4GB GPUs where holding both SoA and AoS buffers is impossible.
    if (use_soa_layout_ && d_synapse_arrays_) {
        SynapseArrays& gpu_ptrs = const_cast<NetworkCUDA*>(this)->h_synapse_arrays_struct_;
        
        // Reusable temp buffers
        std::vector<float> temp_float(num_synapses_);
        std::vector<int> temp_int(num_synapses_);
        
        #define FETCH_S_FLOAT(field, gpu_ptr) \
            cudaMemcpy(temp_float.data(), gpu_ptr, num_synapses_ * sizeof(float), cudaMemcpyDeviceToHost); \
            for(size_t i=0; i<num_synapses_; ++i) host_synapses[i].field = temp_float[i];

        #define FETCH_S_INT(field, gpu_ptr) \
            cudaMemcpy(temp_int.data(), gpu_ptr, num_synapses_ * sizeof(int), cudaMemcpyDeviceToHost); \
            for(size_t i=0; i<num_synapses_; ++i) host_synapses[i].field = temp_int[i];

        // Connectivity
        FETCH_S_INT(pre_neuron_idx, gpu_ptrs.pre_neuron_idx);
        FETCH_S_INT(post_neuron_idx, gpu_ptrs.post_neuron_idx);
        FETCH_S_INT(post_compartment, gpu_ptrs.post_compartment);
        FETCH_S_INT(active, gpu_ptrs.active);
        
        // Weights & Plasticity
        FETCH_S_FLOAT(weight, gpu_ptrs.weight);
        FETCH_S_FLOAT(max_weight, gpu_ptrs.max_weight);
        FETCH_S_FLOAT(min_weight, gpu_ptrs.min_weight);
        FETCH_S_FLOAT(effective_weight, gpu_ptrs.effective_weight);
        FETCH_S_FLOAT(eligibility_trace, gpu_ptrs.eligibility_trace);
        FETCH_S_FLOAT(learning_rate, gpu_ptrs.learning_rate);
        
        // Timing
        FETCH_S_FLOAT(last_pre_spike_time, gpu_ptrs.last_pre_spike_time);
        FETCH_S_FLOAT(last_post_spike_time, gpu_ptrs.last_post_spike_time);
        
        // Modulation
        FETCH_S_FLOAT(dopamine_sensitivity, gpu_ptrs.dopamine_sensitivity);
        FETCH_S_FLOAT(dopamine_level, gpu_ptrs.dopamine_level);
        
        // Short-term Plasticity
        FETCH_S_FLOAT(release_probability, gpu_ptrs.release_probability);
        FETCH_S_FLOAT(facilitation_factor, gpu_ptrs.facilitation_factor);
        FETCH_S_FLOAT(depression_factor, gpu_ptrs.depression_factor);
        
        // Misc
        FETCH_S_FLOAT(presynaptic_calcium, gpu_ptrs.presynaptic_calcium);
        FETCH_S_FLOAT(postsynaptic_calcium, gpu_ptrs.postsynaptic_calcium);
        FETCH_S_FLOAT(delay, gpu_ptrs.delay);
        
        #undef FETCH_S_FLOAT
        #undef FETCH_S_INT
        
    } else if (d_synapses_) {
        cudaMemcpy(host_synapses.data(), d_synapses_, num_synapses_ * sizeof(GPUSynapse), cudaMemcpyDeviceToHost);
    }

    return host_synapses;
}

bool NetworkCUDA::setNeuronStates(const std::vector<GPUNeuronState>& neurons) {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);

    if (!is_initialized_) return false;
    if (neurons.size() != num_neurons_) return false;

    // Just like getting states, we must populate SoA arrays directly to avoid AoS buffer allocation
    if (use_soa_layout_ && d_neuron_arrays_) {
        NeuronArrays& gpu_ptrs = h_neuron_arrays_struct_;
        std::vector<float> temp_float(num_neurons_);
        std::vector<int> temp_int(num_neurons_);
        
        #define PUSH_FLOAT(field, gpu_ptr) \
            for(size_t i=0; i<num_neurons_; ++i) temp_float[i] = neurons[i].field; \
            cudaMemcpy(gpu_ptr, temp_float.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);

        #define PUSH_INT(field, gpu_ptr) \
            for(size_t i=0; i<num_neurons_; ++i) temp_int[i] = neurons[i].field; \
            cudaMemcpy(gpu_ptr, temp_int.data(), num_neurons_ * sizeof(int), cudaMemcpyHostToDevice);

        PUSH_FLOAT(V, gpu_ptrs.V);
        PUSH_FLOAT(u, gpu_ptrs.u);
        PUSH_FLOAT(I_ext, gpu_ptrs.I_ext);
        
        // Synaptic inputs
        for(size_t i=0; i<num_neurons_; ++i) temp_float[i] = neurons[i].I_syn[0];
        cudaMemcpy(gpu_ptrs.I_syn_0, temp_float.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
        
        PUSH_FLOAT(last_spike_time, gpu_ptrs.last_spike_time);
        PUSH_FLOAT(threshold, gpu_ptrs.threshold);
        PUSH_INT(active, gpu_ptrs.active);
        // ... (Add other critical fields as needed for restoration) ...
        
        #undef PUSH_FLOAT
        #undef PUSH_INT
        
    } else if (d_neurons_) {
        cudaMemcpy(d_neurons_, neurons.data(), neurons.size() * sizeof(GPUNeuronState), cudaMemcpyHostToDevice);
    }

    return true;
}

bool NetworkCUDA::setSynapseStates(const std::vector<GPUSynapse>& synapses) {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);

    if (!is_initialized_) return false;
    if (synapses.size() != num_synapses_) return false;

    if (use_soa_layout_ && d_synapse_arrays_) {
        SynapseArrays& gpu_ptrs = h_synapse_arrays_struct_;
        std::vector<float> temp_float(num_synapses_);
        std::vector<int> temp_int(num_synapses_);
        
        #define PUSH_S_FLOAT(field, gpu_ptr) \
            for(size_t i=0; i<num_synapses_; ++i) temp_float[i] = synapses[i].field; \
            cudaMemcpy(gpu_ptr, temp_float.data(), num_synapses_ * sizeof(float), cudaMemcpyHostToDevice);

        #define PUSH_S_INT(field, gpu_ptr) \
            for(size_t i=0; i<num_synapses_; ++i) temp_int[i] = synapses[i].field; \
            cudaMemcpy(gpu_ptr, temp_int.data(), num_synapses_ * sizeof(int), cudaMemcpyHostToDevice);

        PUSH_S_FLOAT(weight, gpu_ptrs.weight);
        PUSH_S_INT(pre_neuron_idx, gpu_ptrs.pre_neuron_idx);
        PUSH_S_INT(post_neuron_idx, gpu_ptrs.post_neuron_idx);
        PUSH_S_FLOAT(eligibility_trace, gpu_ptrs.eligibility_trace);
        // ... (Add other critical fields as needed for restoration) ...
        
        #undef PUSH_S_FLOAT
        #undef PUSH_S_INT
        
    } else if (d_synapses_) {
        cudaMemcpy(d_synapses_, synapses.data(), synapses.size() * sizeof(GPUSynapse), cudaMemcpyHostToDevice);
    }

    return true;
}

std::vector<float> NetworkCUDA::getSynapticWeights() const {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    if (!is_initialized_) return {};
    
    std::vector<float> weights(num_synapses_);
    
    if (use_soa_layout_ && d_synapse_arrays_) {
        // Fast path: copy single array directly
        cudaMemcpy(weights.data(), h_synapse_arrays_struct_.weight, 
                   num_synapses_ * sizeof(float), cudaMemcpyDeviceToHost);
    } else if (d_synapses_) {
        // Slow path: fetch all structs and extract
        auto synapses = getSynapseStates();
        for(size_t i=0; i<num_synapses_; ++i) {
            weights[i] = synapses[i].weight;
        }
    }
    
    return weights;
}

// ============================================================================
// UTILITY AND HELPER METHODS
// ============================================================================

bool NetworkCUDA::checkCudaError(const std::string& operation) const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        total_cuda_errors_++;
        last_cuda_error_ = cudaGetErrorString(error);
        return false;
    }
    return true;
}

void NetworkCUDA::cleanupGPUResources() {
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    
    // Cleanup learning state
    cleanupLearningStateGPU();
    
    // Free SoA arrays
    freeSOAArrays();
    
    // Free neural network memory
    if (d_neurons_) { cudaFree(d_neurons_); d_neurons_ = nullptr; }
    if (d_synapses_) { cudaFree(d_synapses_); d_synapses_ = nullptr; }
    if (d_inputs_) { cudaFree(d_inputs_); d_inputs_ = nullptr; }
    if (d_outputs_) { cudaFree(d_outputs_); d_outputs_ = nullptr; }
    if (d_output_counts_) { cudaFree(d_output_counts_); d_output_counts_ = nullptr; }
    
    // Free working buffers
    if (d_temp_buffer_) { cudaFree(d_temp_buffer_); d_temp_buffer_ = nullptr; }
    if (d_reduction_buffer_) { cudaFree(d_reduction_buffer_); d_reduction_buffer_ = nullptr; }
    if (d_consolidated_count_) { cudaFree(d_consolidated_count_); d_consolidated_count_ = nullptr; }
    
    // Destroy CUDA streams
    if (default_stream_) { cudaStreamDestroy(default_stream_); default_stream_ = nullptr; }
    for (auto stream : compute_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    for (auto stream : memory_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    compute_streams_.clear();
    memory_streams_.clear();
    
    // Destroy CUDA libraries
    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    if (curand_generator_) { curandDestroyGenerator(curand_generator_); curand_generator_ = nullptr; }
    
    // Destroy memory pool
    if (memory_pool_) { cudaMemPoolDestroy(memory_pool_); memory_pool_ = nullptr; }
    
    // Destroy captured graph
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (computation_graph_) { cudaGraphDestroy(computation_graph_); computation_graph_ = nullptr; }
    
    is_initialized_ = false;
}

void NetworkCUDA::warmupGPU() {
    if (!is_initialized_) return;
    
    std::cout << "🔥 Warming up GPU..." << std::endl;
    
    // Run a few dummy computations to warm up the GPU
    for (int i = 0; i < 3; ++i) {
        std::vector<float> dummy_input(num_inputs_, 0.1f);
        processInput(dummy_input);
    }
    
    // Synchronize and measure baseline performance
    cudaDeviceSynchronize();
    
    std::cout << "✅ GPU warmup completed" << std::endl;
}

size_t NetworkCUDA::calculateLearningStateBufferSize() const {
    size_t size = 0;
    
    if (is_initialized_) {
        // Eligibility traces and synaptic tags
        size += num_synapses_ * 3 * sizeof(float);
        
        // Neuromodulator levels and histories
        size += num_neurons_ * (3 + 1000 + 100) * sizeof(float);
        
        // Learning parameters and performance tracking
        size += num_neurons_ * (2 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t));
        
        // Module assignments and boundaries
        size += num_neurons_ * sizeof(int) + 32 * sizeof(int);
        
        // Inter-module connection state
        size += 100 * (7 * sizeof(float) + sizeof(uint64_t) + 2 * sizeof(int)); // Max 100 connections
        size += 100 * (1000 + 1000 + 100) * sizeof(float); // Trace buffers
    }
    
    return size;
}

float NetworkCUDA::calculateRewardPredictionErrorCPU(float actual_reward) {
    // Calculate expected reward as running average
    if (reward_history_count_ < MAX_REWARD_HISTORY) {
        reward_history_sum_ += actual_reward;
        reward_history_count_++;
    } else {
        // Keep rolling average
        expected_reward_ = reward_history_sum_ / reward_history_count_;
        reward_history_sum_ = expected_reward_ * (MAX_REWARD_HISTORY - 1) + actual_reward;
    }
    
    // Update expected reward (exponential moving average)
    float alpha = 0.1f; // Learning rate for expectation
    expected_reward_ = (1.0f - alpha) * expected_reward_ + alpha * actual_reward;
    
    // Calculate reward prediction error (RPE)
    float rpe = actual_reward - expected_reward_;
    
    // Optional: Use CUB for parallel reductions if we need neuron-level stats
    // For now, simple CPU calculation is sufficient and avoids GPU serialization
    
    return rpe;
}

void NetworkCUDA::freeSOAArrays() {
    if (d_neuron_arrays_) {
        // Copy structure from device to get pointers
        NeuronArrays h_arrays;
        cudaMemcpy(&h_arrays, d_neuron_arrays_, sizeof(NeuronArrays), cudaMemcpyDeviceToHost);
        
        // Free all allocated arrays
        if (h_arrays.V) cudaFree(h_arrays.V);
        if (h_arrays.u) cudaFree(h_arrays.u);
        if (h_arrays.I_syn_0) cudaFree(h_arrays.I_syn_0);
        if (h_arrays.I_syn_1) cudaFree(h_arrays.I_syn_1);
        if (h_arrays.I_syn_2) cudaFree(h_arrays.I_syn_2);
        if (h_arrays.I_syn_3) cudaFree(h_arrays.I_syn_3);
        if (h_arrays.I_ext) cudaFree(h_arrays.I_ext);
        if (h_arrays.ca_conc_0) cudaFree(h_arrays.ca_conc_0);
        if (h_arrays.ca_conc_1) cudaFree(h_arrays.ca_conc_1);
        if (h_arrays.ca_conc_2) cudaFree(h_arrays.ca_conc_2);
        if (h_arrays.ca_conc_3) cudaFree(h_arrays.ca_conc_3);
        if (h_arrays.last_spike_time) cudaFree(h_arrays.last_spike_time);
        if (h_arrays.previous_spike_time) cudaFree(h_arrays.previous_spike_time);
        if (h_arrays.average_firing_rate) cudaFree(h_arrays.average_firing_rate);
        if (h_arrays.average_activity) cudaFree(h_arrays.average_activity);
        if (h_arrays.activity_level) cudaFree(h_arrays.activity_level);
        if (h_arrays.firing_rate) cudaFree(h_arrays.firing_rate);
        if (h_arrays.excitability) cudaFree(h_arrays.excitability);
        if (h_arrays.synaptic_scaling_factor) cudaFree(h_arrays.synaptic_scaling_factor);
        if (h_arrays.bcm_threshold) cudaFree(h_arrays.bcm_threshold);
        if (h_arrays.plasticity_threshold) cudaFree(h_arrays.plasticity_threshold);
        if (h_arrays.threshold) cudaFree(h_arrays.threshold);
        if (h_arrays.dopamine_concentration) cudaFree(h_arrays.dopamine_concentration);
        if (h_arrays.acetylcholine_level) cudaFree(h_arrays.acetylcholine_level);
        if (h_arrays.serotonin_level) cudaFree(h_arrays.serotonin_level);
        if (h_arrays.norepinephrine_level) cudaFree(h_arrays.norepinephrine_level);
        if (h_arrays.na_m) cudaFree(h_arrays.na_m);
        if (h_arrays.na_h) cudaFree(h_arrays.na_h);
        if (h_arrays.k_n) cudaFree(h_arrays.k_n);
        if (h_arrays.ca_channel_state) cudaFree(h_arrays.ca_channel_state);
        if (h_arrays.neuron_type) cudaFree(h_arrays.neuron_type);
        if (h_arrays.layer_id) cudaFree(h_arrays.layer_id);
        if (h_arrays.column_id) cudaFree(h_arrays.column_id);
        if (h_arrays.active) cudaFree(h_arrays.active);
        if (h_arrays.energy_level) cudaFree(h_arrays.energy_level);
        if (h_arrays.metabolic_demand) cudaFree(h_arrays.metabolic_demand);
        
        cudaFree(d_neuron_arrays_);
        d_neuron_arrays_ = nullptr;
    }
    
    if (d_synapse_arrays_) {
        // Copy structure from device to get pointers
        SynapseArrays h_arrays;
        cudaMemcpy(&h_arrays, d_synapse_arrays_, sizeof(SynapseArrays), cudaMemcpyDeviceToHost);
        
        // Free all allocated arrays
        if (h_arrays.pre_neuron_idx) cudaFree(h_arrays.pre_neuron_idx);
        if (h_arrays.post_neuron_idx) cudaFree(h_arrays.post_neuron_idx);
        if (h_arrays.post_compartment) cudaFree(h_arrays.post_compartment);
        if (h_arrays.active) cudaFree(h_arrays.active);
        if (h_arrays.weight) cudaFree(h_arrays.weight);
        if (h_arrays.max_weight) cudaFree(h_arrays.max_weight);
        if (h_arrays.min_weight) cudaFree(h_arrays.min_weight);
        if (h_arrays.effective_weight) cudaFree(h_arrays.effective_weight);
        if (h_arrays.eligibility_trace) cudaFree(h_arrays.eligibility_trace);
        if (h_arrays.learning_rate) cudaFree(h_arrays.learning_rate);
        if (h_arrays.last_pre_spike_time) cudaFree(h_arrays.last_pre_spike_time);
        if (h_arrays.last_post_spike_time) cudaFree(h_arrays.last_post_spike_time);
        if (h_arrays.dopamine_sensitivity) cudaFree(h_arrays.dopamine_sensitivity);
        if (h_arrays.dopamine_level) cudaFree(h_arrays.dopamine_level);
        if (h_arrays.release_probability) cudaFree(h_arrays.release_probability);
        if (h_arrays.facilitation_factor) cudaFree(h_arrays.facilitation_factor);
        if (h_arrays.depression_factor) cudaFree(h_arrays.depression_factor);
        if (h_arrays.presynaptic_calcium) cudaFree(h_arrays.presynaptic_calcium);
        if (h_arrays.postsynaptic_calcium) cudaFree(h_arrays.postsynaptic_calcium);
        if (h_arrays.plasticity_modulation) cudaFree(h_arrays.plasticity_modulation);
        if (h_arrays.metaplasticity_factor) cudaFree(h_arrays.metaplasticity_factor);
        if (h_arrays.delay) cudaFree(h_arrays.delay);
        if (h_arrays.homeostatic_scaling) cudaFree(h_arrays.homeostatic_scaling);
        if (h_arrays.receptor_index) cudaFree(h_arrays.receptor_index);
        if (h_arrays.vesicle_count) cudaFree(h_arrays.vesicle_count);
        
        cudaFree(d_synapse_arrays_);
        d_synapse_arrays_ = nullptr;
    }
}

// CUDA kernel to convert AoS neurons to SoA
__global__ void convertNeuronsAoSToSoA_kernel(
    const GPUNeuronState* aos,
    NeuronArrays arrays,
    int num_neurons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = aos[idx];
    
    // Core membrane dynamics
    arrays.V[idx] = neuron.V;
    arrays.u[idx] = neuron.u;
    arrays.I_syn_0[idx] = neuron.I_syn[0];
    arrays.I_syn_1[idx] = neuron.I_syn[1];
    arrays.I_syn_2[idx] = neuron.I_syn[2];
    arrays.I_syn_3[idx] = neuron.I_syn[3];
    arrays.I_ext[idx] = neuron.I_ext;
    
    // Calcium dynamics
    arrays.ca_conc_0[idx] = neuron.ca_conc[0];
    arrays.ca_conc_1[idx] = neuron.ca_conc[1];
    arrays.ca_conc_2[idx] = neuron.ca_conc[2];
    arrays.ca_conc_3[idx] = neuron.ca_conc[3];
    
    // Timing
    arrays.last_spike_time[idx] = neuron.last_spike_time;
    arrays.previous_spike_time[idx] = neuron.previous_spike_time;
    
    // Activity
    arrays.average_firing_rate[idx] = neuron.average_firing_rate;
    arrays.average_activity[idx] = neuron.average_activity;
    arrays.activity_level[idx] = neuron.activity_level;
    arrays.firing_rate[idx] = neuron.firing_rate;
    
    // Plasticity
    arrays.excitability[idx] = neuron.excitability;
    arrays.synaptic_scaling_factor[idx] = neuron.synaptic_scaling_factor;
    arrays.bcm_threshold[idx] = neuron.bcm_threshold;
    arrays.plasticity_threshold[idx] = neuron.plasticity_threshold;
    arrays.threshold[idx] = neuron.threshold;
    
    // Neuromodulation
    arrays.dopamine_concentration[idx] = neuron.dopamine_concentration;
    arrays.acetylcholine_level[idx] = neuron.acetylcholine_level;
    arrays.serotonin_level[idx] = neuron.serotonin_level;
    arrays.norepinephrine_level[idx] = neuron.norepinephrine_level;
    
    // Ion channels
    arrays.na_m[idx] = neuron.na_m;
    arrays.na_h[idx] = neuron.na_h;
    arrays.k_n[idx] = neuron.k_n;
    arrays.ca_channel_state[idx] = neuron.ca_channel_state;
    
    // Network properties
    arrays.neuron_type[idx] = neuron.neuron_type;
    arrays.layer_id[idx] = neuron.layer_id;
    arrays.column_id[idx] = neuron.column_id;
    arrays.active[idx] = neuron.active;
    
    // Metabolism
    arrays.energy_level[idx] = neuron.energy_level;
    arrays.metabolic_demand[idx] = neuron.metabolic_demand;
}

// CUDA kernel to convert SoA neurons back to AoS
__global__ void convertNeuronsSoAToAoS_kernel(
    const NeuronArrays arrays,
    GPUNeuronState* aos,
    int num_neurons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = aos[idx];
    
    // Core membrane dynamics
    neuron.V = arrays.V[idx];
    neuron.u = arrays.u[idx];
    neuron.I_syn[0] = arrays.I_syn_0[idx];
    neuron.I_syn[1] = arrays.I_syn_1[idx];
    neuron.I_syn[2] = arrays.I_syn_2[idx];
    neuron.I_syn[3] = arrays.I_syn_3[idx];
    neuron.I_ext = arrays.I_ext[idx];
    
    // Calcium dynamics
    neuron.ca_conc[0] = arrays.ca_conc_0[idx];
    neuron.ca_conc[1] = arrays.ca_conc_1[idx];
    neuron.ca_conc[2] = arrays.ca_conc_2[idx];
    neuron.ca_conc[3] = arrays.ca_conc_3[idx];
    
    // Timing
    neuron.last_spike_time = arrays.last_spike_time[idx];
    neuron.previous_spike_time = arrays.previous_spike_time[idx];
    
    // Activity
    neuron.average_firing_rate = arrays.average_firing_rate[idx];
    neuron.average_activity = arrays.average_activity[idx];
    neuron.activity_level = arrays.activity_level[idx];
    neuron.firing_rate = arrays.firing_rate[idx];
    
    // Plasticity
    neuron.excitability = arrays.excitability[idx];
    neuron.synaptic_scaling_factor = arrays.synaptic_scaling_factor[idx];
    neuron.bcm_threshold = arrays.bcm_threshold[idx];
    neuron.plasticity_threshold = arrays.plasticity_threshold[idx];
    neuron.threshold = arrays.threshold[idx];
    
    // Neuromodulation
    neuron.dopamine_concentration = arrays.dopamine_concentration[idx];
    neuron.acetylcholine_level = arrays.acetylcholine_level[idx];
    neuron.serotonin_level = arrays.serotonin_level[idx];
    neuron.norepinephrine_level = arrays.norepinephrine_level[idx];
    
    // Ion channels
    neuron.na_m = arrays.na_m[idx];
    neuron.na_h = arrays.na_h[idx];
    neuron.k_n = arrays.k_n[idx];
    neuron.ca_channel_state = arrays.ca_channel_state[idx];
    
    // Network properties
    neuron.neuron_type = arrays.neuron_type[idx];
    neuron.layer_id = arrays.layer_id[idx];
    neuron.column_id = arrays.column_id[idx];
    neuron.active = arrays.active[idx];
    
    // Metabolism
    neuron.energy_level = arrays.energy_level[idx];
    neuron.metabolic_demand = arrays.metabolic_demand[idx];
}

// CUDA kernel to convert AoS synapses to SoA
__global__ void convertSynapsesAoSToSoA_kernel(
    const GPUSynapse* aos,
    SynapseArrays arrays,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    const GPUSynapse& synapse = aos[idx];
    
    // Connectivity
    arrays.pre_neuron_idx[idx] = synapse.pre_neuron_idx;
    arrays.post_neuron_idx[idx] = synapse.post_neuron_idx;
    arrays.post_compartment[idx] = synapse.post_compartment;
    arrays.active[idx] = synapse.active;
    
    // Synaptic properties
    arrays.weight[idx] = synapse.weight;
    arrays.max_weight[idx] = synapse.max_weight;
    arrays.min_weight[idx] = synapse.min_weight;
    arrays.effective_weight[idx] = synapse.effective_weight;
    
    // Plasticity
    arrays.eligibility_trace[idx] = synapse.eligibility_trace;
    arrays.learning_rate[idx] = synapse.learning_rate;
    
    // Timing
    arrays.last_pre_spike_time[idx] = synapse.last_pre_spike_time;
    arrays.last_post_spike_time[idx] = synapse.last_post_spike_time;
    
    // Neuromodulation
    arrays.dopamine_sensitivity[idx] = synapse.dopamine_sensitivity;
    arrays.dopamine_level[idx] = synapse.dopamine_level;
    
    // Short-term plasticity
    arrays.release_probability[idx] = synapse.release_probability;
    arrays.facilitation_factor[idx] = synapse.facilitation_factor;
    arrays.depression_factor[idx] = synapse.depression_factor;
    
    // Calcium dynamics
    arrays.presynaptic_calcium[idx] = synapse.presynaptic_calcium;
    arrays.postsynaptic_calcium[idx] = synapse.postsynaptic_calcium;
    
    // Other properties
    arrays.plasticity_modulation[idx] = synapse.plasticity_modulation;
    arrays.metaplasticity_factor[idx] = synapse.metaplasticity_factor;
    arrays.delay[idx] = synapse.delay;
    arrays.homeostatic_scaling[idx] = synapse.homeostatic_scaling;
    arrays.receptor_index[idx] = synapse.receptor_index;
    arrays.vesicle_count[idx] = synapse.vesicle_count;
}

// CUDA kernel to convert SoA synapses back to AoS
__global__ void convertSynapsesSoAToAoS_kernel(
    const SynapseArrays arrays,
    GPUSynapse* aos,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = aos[idx];
    
    // Connectivity
    synapse.pre_neuron_idx = arrays.pre_neuron_idx[idx];
    synapse.post_neuron_idx = arrays.post_neuron_idx[idx];
    synapse.post_compartment = arrays.post_compartment[idx];
    synapse.active = arrays.active[idx];
    
    // Synaptic properties
    synapse.weight = arrays.weight[idx];
    synapse.max_weight = arrays.max_weight[idx];
    synapse.min_weight = arrays.min_weight[idx];
    synapse.effective_weight = arrays.effective_weight[idx];
    
    // Plasticity
    synapse.eligibility_trace = arrays.eligibility_trace[idx];
    synapse.learning_rate = arrays.learning_rate[idx];
    
    // Timing
    synapse.last_pre_spike_time = arrays.last_pre_spike_time[idx];
    synapse.last_post_spike_time = arrays.last_post_spike_time[idx];
    
    // Neuromodulation
    synapse.dopamine_sensitivity = arrays.dopamine_sensitivity[idx];
    synapse.dopamine_level = arrays.dopamine_level[idx];
    
    // Short-term plasticity
    synapse.release_probability = arrays.release_probability[idx];
    synapse.facilitation_factor = arrays.facilitation_factor[idx];
    synapse.depression_factor = arrays.depression_factor[idx];
    
    // Calcium dynamics
    synapse.presynaptic_calcium = arrays.presynaptic_calcium[idx];
    synapse.postsynaptic_calcium = arrays.postsynaptic_calcium[idx];
    
    // Other properties
    synapse.plasticity_modulation = arrays.plasticity_modulation[idx];
    synapse.metaplasticity_factor = arrays.metaplasticity_factor[idx];
    synapse.delay = arrays.delay[idx];
    synapse.homeostatic_scaling = arrays.homeostatic_scaling[idx];
    synapse.receptor_index = arrays.receptor_index[idx];
    synapse.vesicle_count = arrays.vesicle_count[idx];
}

void NetworkCUDA::convertAoSToSoA() {
    if (!d_neurons_ || !d_synapses_ || !d_neuron_arrays_ || !d_synapse_arrays_) return;
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), void());
    
    // Using previously defined kernels
    const int threads = 256;
    convertNeuronsAoSToSoA_kernel<<<(num_neurons_+threads-1)/threads, threads>>>(d_neurons_, h_neuron_arrays_struct_, num_neurons_);
    convertSynapsesAoSToSoA_kernel<<<(num_synapses_+threads-1)/threads, threads>>>(d_synapses_, h_synapse_arrays_struct_, num_synapses_);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), void());
}

void NetworkCUDA::convertSoAToAoS() {
    if (!d_neurons_ || !d_synapses_ || !d_neuron_arrays_ || !d_synapse_arrays_) return;
    std::lock_guard<std::recursive_mutex> lock(cuda_mutex_);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), void());
    
    const int threads = 256;
    convertNeuronsSoAToAoS_kernel<<<(num_neurons_+threads-1)/threads, threads>>>(h_neuron_arrays_struct_, d_neurons_, num_neurons_);
    convertSynapsesSoAToAoS_kernel<<<(num_synapses_+threads-1)/threads, threads>>>(h_synapse_arrays_struct_, d_synapses_, num_synapses_);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), void());
}

// ============================================================================
// CUDA UTILITY FUNCTIONS
// ============================================================================

bool isCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

int getCudaDeviceCount() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

int getOptimalCudaDevice() {
    int device_count = getCudaDeviceCount();
    if (device_count == 0) return -1;
    
    int best_device = 0;
    size_t max_memory = 0;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.totalGlobalMem > max_memory && prop.major >= 3) { // Require compute capability 3.0+
                max_memory = prop.totalGlobalMem;
                best_device = i;
            }
        }
    }
    
    return best_device;
}

std::pair<size_t, size_t> getCudaMemoryInfo(int device_id) {
    size_t free_bytes = 0, total_bytes = 0;
    
    int current_device;
    cudaGetDevice(&current_device);
    
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (device_id >= 0) {
        cudaSetDevice(current_device);
    }
    
    return {free_bytes, total_bytes};
}

void warmUpCudaDevice(int device_id) {
    int current_device;
    cudaGetDevice(&current_device);
    
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    
    // Allocate and free a small amount of memory to initialize the context
    void* temp_ptr;
    cudaMalloc(&temp_ptr, 1024);
    cudaFree(temp_ptr);
    
    // Run a simple kernel to warm up the device
    cudaDeviceSynchronize();
    
    if (device_id >= 0) {
        cudaSetDevice(current_device);
    }
}

std::string getCudaRuntimeVersion() {
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string getCudaDriverVersion() {
    int driver_version;
    cudaDriverGetVersion(&driver_version);
    
    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

// ============================================================================
// MISSING HELPER METHODS
// ============================================================================

bool NetworkCUDA::shouldSynchronize() const {
    // Simple heuristic: synchronize every 100 updates or if there's significant learning activity
    static int update_count = 0;
    update_count++;
    
    if (update_count % 100 == 0) {
        return true;
    }
    
    // Check if there's significant learning activity
    if (performance_metrics_.last_update_time_ms > performance_metrics_.avg_update_time_ms * 1.5f) {
        return true;
    }
    
    return false;
}

float NetworkCUDA::calculateFragmentationRatio() const {
    // Simple fragmentation calculation based on memory usage patterns
    // In a real implementation, this would analyze memory allocation patterns
    
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0.0f;
    }
    
    size_t used_bytes = total_bytes - free_bytes;
    if (total_bytes == 0) {
        return 0.0f;
    }
    
    // Simple heuristic: fragmentation increases with memory usage
    float usage_ratio = static_cast<float>(used_bytes) / total_bytes;
    
    // Assume fragmentation is minimal below 50% usage, increases after that
    if (usage_ratio < 0.5f) {
        return usage_ratio * 0.1f; // Low fragmentation
    } else {
        return 0.05f + (usage_ratio - 0.5f) * 0.4f; // Increasing fragmentation
    }
}

bool NetworkCUDA::allocateWorkingBuffers() {
    try {
        // Allocate temporary computation buffer
        size_t temp_buffer_size = std::max(num_neurons_, num_synapses_) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_temp_buffer_, temp_buffer_size));
        
        // Allocate reduction buffer for parallel reductions
        size_t reduction_buffer_size = 1024 * sizeof(float); // For reduction operations
        CUDA_CHECK(cudaMalloc(&d_reduction_buffer_, reduction_buffer_size));
        
        // Allocate counter for consolidation operations
        CUDA_CHECK(cudaMalloc(&d_consolidated_count_, sizeof(int)));
        
        std::cout << "💾 Allocated working buffers:" << std::endl;
        std::cout << "   Temp buffer: " << temp_buffer_size / (1024*1024) << " MB" << std::endl;
        std::cout << "   Reduction buffer: " << reduction_buffer_size / 1024 << " KB" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating working buffers: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::initializeNeuralNetworkData() {
    try {
        // Initialize neurons with default values
        if (d_neurons_) {
            std::vector<GPUNeuronState> initial_neurons(num_neurons_);
            
            // Zero-initialize all neurons first to avoid garbage values
            std::memset(initial_neurons.data(), 0, num_neurons_ * sizeof(GPUNeuronState));

            for (size_t i = 0; i < num_neurons_; ++i) {
                // Core membrane dynamics
                initial_neurons[i].V = -65.0f; // Resting potential
                initial_neurons[i].u = 0.0f; // Recovery variable
                for (int j = 0; j < MAX_COMPARTMENTS; ++j) {
                    initial_neurons[i].I_syn[j] = 0.0f;
                    initial_neurons[i].ca_conc[j] = 50.0e-9f; // Resting calcium
                }
                initial_neurons[i].I_ext = 0.0f;

                // Timing and activity
                initial_neurons[i].last_spike_time = -1000.0f; // Far in the past
                initial_neurons[i].previous_spike_time = -1000.0f;
                initial_neurons[i].average_firing_rate = 0.0f;

                // Plasticity and adaptation
                initial_neurons[i].excitability = 1.0f; // Normal excitability
                initial_neurons[i].synaptic_scaling_factor = 1.0f;
                initial_neurons[i].threshold = -50.0f; // Firing threshold

                // Network properties
                initial_neurons[i].active = 1; // Set as active
                initial_neurons[i].neuron_type = (i % 5 == 0) ? 0 : 1; // 20% inhibitory, 80% excitatory
            }

            CUDA_CHECK(cudaMemcpy(d_neurons_, initial_neurons.data(),
                                 num_neurons_ * sizeof(GPUNeuronState),
                                 cudaMemcpyHostToDevice));
        }
        
        // Initialize synapses with random weights
        if (d_synapses_) {
            std::vector<GPUSynapse> initial_synapses(num_synapses_);

            // Zero-initialize all synapses first
            std::memset(initial_synapses.data(), 0, num_synapses_ * sizeof(GPUSynapse));

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> weight_dist(0.0f, 0.1f);

            for (size_t i = 0; i < num_synapses_; ++i) {
                // Connectivity
                initial_synapses[i].pre_neuron_idx = static_cast<int>(i % num_neurons_);
                initial_synapses[i].post_neuron_idx = static_cast<int>((i + 1) % num_neurons_);
                initial_synapses[i].active = 1; // Set as active

                // Synaptic properties
                initial_synapses[i].weight = weight_dist(gen);
                initial_synapses[i].max_weight = 1.0f;
                initial_synapses[i].min_weight = 0.0f;
                initial_synapses[i].effective_weight = initial_synapses[i].weight;
                
                // Plasticity
                initial_synapses[i].is_plastic = true; // Enable plasticity
                initial_synapses[i].eligibility_trace = 0.0f;
                initial_synapses[i].learning_rate = 0.01f;
                
                // Timing
                initial_synapses[i].last_pre_spike_time = -1000.0f;
                initial_synapses[i].last_post_spike_time = -1000.0f;
                
                // Vesicle dynamics
                initial_synapses[i].vesicle_count = 100;
                initial_synapses[i].release_probability = 0.5f;
            }

            CUDA_CHECK(cudaMemcpy(d_synapses_, initial_synapses.data(),
                                 num_synapses_ * sizeof(GPUSynapse),
                                 cudaMemcpyHostToDevice));
                                 
            // Ensure copies are complete
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Convert AoS to SoA for optimized computation
        if (use_soa_layout_ && d_neuron_arrays_ && d_synapse_arrays_) {
            convertAoSToSoA();
            std::cout << "✓ Converted neural data to SoA layout" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error initializing neural network data: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::initializeLearningStateData() {
    // This method would initialize the learning state data structures
    // For now, return true as the allocation already initializes to defaults
    return true;
}

void NetworkCUDA::updateMemoryStats() const {
    // Update memory statistics - this is called internally
    // The actual implementation is in getMemoryStats()
}

void NetworkCUDA::cleanupLearningStateGPU() {
    if (d_learning_state_) {
        // Free learning state arrays
        GPULearningState h_learning_state;
        if (cudaMemcpy(&h_learning_state, d_learning_state_, sizeof(GPULearningState), cudaMemcpyDeviceToHost) == cudaSuccess) {
            if (h_learning_state.eligibility_traces) cudaFree(h_learning_state.eligibility_traces);
            if (h_learning_state.synaptic_tags) cudaFree(h_learning_state.synaptic_tags);
            if (h_learning_state.consolidation_weights) cudaFree(h_learning_state.consolidation_weights);
            if (h_learning_state.neuromodulator_levels) cudaFree(h_learning_state.neuromodulator_levels);
            if (h_learning_state.firing_rate_history) cudaFree(h_learning_state.firing_rate_history);
            if (h_learning_state.prediction_errors) cudaFree(h_learning_state.prediction_errors);
            if (h_learning_state.learning_rates) cudaFree(h_learning_state.learning_rates);
            if (h_learning_state.plasticity_thresholds) cudaFree(h_learning_state.plasticity_thresholds);
            if (h_learning_state.learning_step_counts) cudaFree(h_learning_state.learning_step_counts);
            if (h_learning_state.reward_history) cudaFree(h_learning_state.reward_history);
            if (h_learning_state.history_indices) cudaFree(h_learning_state.history_indices);
            if (h_learning_state.module_assignments) cudaFree(h_learning_state.module_assignments);
            if (h_learning_state.module_boundaries) cudaFree(h_learning_state.module_boundaries);
        }
        
        cudaFree(d_learning_state_);
        d_learning_state_ = nullptr;
    }
    
    if (d_inter_module_state_) {
        // Free inter-module state arrays
        GPUInterModuleState h_inter_module_state;
        if (cudaMemcpy(&h_inter_module_state, d_inter_module_state_, sizeof(GPUInterModuleState), cudaMemcpyDeviceToHost) == cudaSuccess) {
            if (h_inter_module_state.connection_strengths) cudaFree(h_inter_module_state.connection_strengths);
            if (h_inter_module_state.usage_frequencies) cudaFree(h_inter_module_state.usage_frequencies);
            if (h_inter_module_state.correlation_strengths) cudaFree(h_inter_module_state.correlation_strengths);
            if (h_inter_module_state.activation_counts) cudaFree(h_inter_module_state.activation_counts);
            if (h_inter_module_state.pre_synaptic_traces) cudaFree(h_inter_module_state.pre_synaptic_traces);
            if (h_inter_module_state.post_synaptic_traces) cudaFree(h_inter_module_state.post_synaptic_traces);
            if (h_inter_module_state.timing_differences) cudaFree(h_inter_module_state.timing_differences);
            if (h_inter_module_state.source_modules) cudaFree(h_inter_module_state.source_modules);
            if (h_inter_module_state.target_modules) cudaFree(h_inter_module_state.target_modules);
        }
        
        cudaFree(d_inter_module_state_);
        d_inter_module_state_ = nullptr;
    }
}