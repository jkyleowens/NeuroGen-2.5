// Update 6: CUDA Kernel Extensions for Learning State
// File: include/NeuroGen/cuda/LearningStateKernels.cuh (NEW FILE)

#ifndef LEARNING_STATE_KERNELS_CUH
#define LEARNING_STATE_KERNELS_CUH

#include "GPUNeuralStructures.h"
#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief GPU structures for learning state management
 */
struct GPULearningState {
    float* eligibility_traces;      // Per-synapse eligibility traces
    float* synaptic_tags;          // Per-synapse tagging markers
    float* neuromodulator_levels;   // Per-neuron neuromodulator concentrations
    float* firing_rate_history;    // Per-neuron firing rate buffer
    float* prediction_errors;      // Per-neuron prediction error history
    
    // Learning parameters
    float* learning_rates;         // Per-neuron learning rates
    float* plasticity_thresholds;  // Per-neuron plasticity thresholds
    float* consolidation_weights;  // Per-synapse consolidation strength
    
    // Performance tracking
    uint64_t* learning_step_counts; // Per-neuron learning step counters
    float* reward_history;         // Per-neuron reward history buffer
    uint32_t* history_indices;     // Circular buffer indices
    
    // Module assignments and boundaries
    int* module_assignments;       // Which module each neuron belongs to
    int* module_boundaries;        // Start/end indices for each module
    int num_modules;
};

/**
 * @brief Inter-module connection state on GPU
 */
struct GPUInterModuleState {
    float* connection_strengths;    // Between-module connection strengths
    float* usage_frequencies;      // How often connections are used
    float* correlation_strengths;  // Hebbian correlation measurements
    uint64_t* activation_counts;   // Number of times each connection activated
    
    // Spike timing buffers
    float* pre_synaptic_traces;   // Pre-synaptic activity traces
    float* post_synaptic_traces;  // Post-synaptic activity traces
    float* timing_differences;   // STDP timing difference buffer
    
    int* source_modules;         // Source module for each connection
    int* target_modules;         // Target module for each connection
    int num_connections;
};

/**
 * @namespace LearningStateKernels
 * @brief CUDA kernels for learning state management
 */
namespace LearningStateKernels {

// ============================================================================
// LEARNING TRACE MANAGEMENT
// ============================================================================

/**
 * @brief Update eligibility traces for all synapses
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param synapses GPU synapse array
 * @param reward_signal Global reward signal
 * @param dt Time step
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void update_eligibility_traces(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float reward_signal,
    float dt,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Apply synaptic tagging based on novelty
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param novelty_signal Novelty/surprise signal
 * @param dt Time step
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void apply_synaptic_tagging(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float novelty_signal,
    float dt,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Update neuromodulator levels
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param dopamine_level Global dopamine level
 * @param acetylcholine_level Global acetylcholine level
 * @param norepinephrine_level Global norepinephrine level
 * @param dt Time step
 * @param num_neurons Total number of neurons
 */
void update_neuromodulators(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    float dopamine_level,
    float acetylcholine_level,
    float norepinephrine_level,
    float dt,
    int num_neurons
);

// ============================================================================
// MEMORY CONSOLIDATION
// ============================================================================

/**
 * @brief Perform memory consolidation across modules
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param synapses GPU synapse array
 * @param consolidation_strength Strength of consolidation (0-1)
 * @param consolidated_count Output: number of synapses consolidated
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void perform_memory_consolidation(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float consolidation_strength,
    int* consolidated_count,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Consolidate specific module
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param synapses GPU synapse array
 * @param module_id Module to consolidate
 * @param consolidation_strength Strength of consolidation
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void consolidate_module(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int module_id,
    float consolidation_strength,
    int num_neurons,
    int num_synapses
);

// ============================================================================
// INTER-MODULE LEARNING
// ============================================================================

/**
 * @brief Update inter-module connection strengths
 * @param inter_module_state GPU inter-module state structure
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param learning_rate_multiplier Global learning rate modifier
 * @param dt Time step
 * @param num_neurons Total number of neurons
 */
void update_inter_module_connections(
    GPUInterModuleState* inter_module_state,
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float learning_rate_multiplier,
    float dt,
    int num_neurons
);

/**
 * @brief Apply Hebbian learning to inter-module connections
 * @param inter_module_state GPU inter-module state structure
 * @param source_activities Source module activity levels
 * @param target_activities Target module activity levels
 * @param learning_rate Learning rate for connection updates
 * @param num_connections Number of inter-module connections
 */
void apply_hebbian_learning(
    GPUInterModuleState* inter_module_state,
    const float* source_activities,
    const float* target_activities,
    float learning_rate,
    int num_connections
);

// ============================================================================
// STATE PERSISTENCE
// ============================================================================

/**
 * @brief Save learning state for specific module to buffer
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param synapses GPU synapse array
 * @param module_id Module to save
 * @param output_buffer Host buffer to copy data to
 * @param buffer_size Size of output buffer
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void save_module_learning_state(
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    const GPUSynapse* synapses,
    int module_id,
    float* output_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Load learning state for specific module from buffer
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param synapses GPU synapse array
 * @param module_id Module to load
 * @param input_buffer Host buffer containing data
 * @param buffer_size Size of input buffer
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void load_module_learning_state(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int module_id,
    const float* input_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Save complete learning state to host memory
 * @param learning_state GPU learning state structure
 * @param inter_module_state GPU inter-module state structure
 * @param host_buffer Host buffer for complete state
 * @param buffer_size Size of host buffer
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void save_complete_learning_state(
    const GPULearningState* learning_state,
    const GPUInterModuleState* inter_module_state,
    uint8_t* host_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Load complete learning state from host memory
 * @param learning_state GPU learning state structure
 * @param inter_module_state GPU inter-module state structure
 * @param host_buffer Host buffer containing complete state
 * @param buffer_size Size of host buffer
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void load_complete_learning_state(
    GPULearningState* learning_state,
    GPUInterModuleState* inter_module_state,
    const uint8_t* host_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
);

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

/**
 * @brief Calculate module performance metrics
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param module_metrics Output buffer for per-module metrics
 * @param num_modules Number of modules
 * @param num_neurons Total number of neurons
 */
void calculate_module_performance(
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float* module_metrics,
    int num_modules,
    int num_neurons
);

/**
 * @brief Update learning statistics
 * @param learning_state GPU learning state structure
 * @param neurons GPU neuron state array
 * @param reward_signal Current reward signal
 * @param prediction_error Current prediction error
 * @param dt Time step
 * @param num_neurons Total number of neurons
 */
void update_learning_statistics(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float reward_signal,
    float prediction_error,
    float dt,
    int num_neurons
);

} // namespace LearningStateKernels

#endif // LEARNING_STATE_KERNELS_CUH