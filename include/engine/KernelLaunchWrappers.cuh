#ifndef KERNEL_LAUNCH_WRAPPERS_CUH
#define KERNEL_LAUNCH_WRAPPERS_CUH

#include "GPUNeuralStructures.h"
#include <cuda_runtime.h>

// Forward declarations for modular architecture support
struct ModuleConfiguration;
struct AttentionState;

/**
 * @namespace KernelLaunchWrappers
 * @brief Centralized wrapper functions for all CUDA kernel launches
 * 
 * This namespace provides a clean interface between the host-side NetworkCUDA class
 * and the device-side CUDA kernels, supporting both traditional neural networks
 * and the new modular architecture with variable-size, self-contained modules.
 */
namespace KernelLaunchWrappers {

// ============================================================================
// CORE NEURAL PROCESSING KERNELS
// ============================================================================

/**
 * @brief Initialize ion channels and basic neuron state
 * @param neurons Device pointer to neuron array
 * @param num_neurons Total number of neurons to initialize
 */
void initialize_ion_channels(GPUNeuronState* neurons, int num_neurons);

/**
 * @brief Update neuron states using Izhikevich dynamics
 * @param neurons Device pointer to neuron array
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_neurons Total number of neurons
 */
void update_neuron_states(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);

/**
 * @brief Fused neuron update (SoA version) - combines update, calcium, neuromodulation
 * @param neuron_arrays Device pointer to neuron SoA structure
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param dopamine_level Global dopamine concentration
 * @param serotonin_level Global serotonin concentration
 * @param num_neurons Total number of neurons
 */
void launch_fused_neuron_update(
    NeuronArrays* neuron_arrays,
    float current_time,
    float dt,
    float dopamine_level,
    float serotonin_level,
    int num_neurons
);

/**
 * @brief Update calcium dynamics for plasticity mechanisms
 * @param neurons Device pointer to neuron array
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_neurons Total number of neurons
 */
void update_calcium_dynamics(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);

/**
 * @brief Process STDP and update eligibility traces
 * @param synapses Device pointer to synapse array
 * @param neurons Device pointer to neuron array (read-only)
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_synapses Total number of synapses
 */
void run_stdp_and_eligibility(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Fused plasticity kernel (SoA version) - combines STDP, eligibility, reward
 * @param synapse_arrays Device pointer to synapse SoA structure
 * @param neuron_arrays Device pointer to neuron SoA structure
 * @param reward_signal Reward prediction error
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_synapses Total number of synapses
 */
void launch_fused_plasticity(
    SynapseArrays* synapse_arrays,
    NeuronArrays* neuron_arrays,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Apply reward signals and neuromodulation
 * @param synapses Device pointer to synapse array
 * @param neurons Device pointer to neuron array
 * @param reward Global reward signal
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_synapses Total number of synapses
 */
void apply_reward_and_adaptation(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Execute homeostatic mechanisms for network stability
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void run_homeostatic_mechanisms(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float current_time,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Compute compact neuron output groups directly on the GPU.
 *
 * @param neurons Device pointer to neuron array.
 * @param output_buffer Device pointer that will hold averaged outputs.
 * @param output_counts Device pointer for per-group counts (temporary).
 * @param num_neurons Total number of neurons.
 * @param num_outputs Total number of output groups.
 * @param group_size Number of neurons mapped to each output group.
 */
void compute_neuron_outputs(
    const GPUNeuronState* neurons,
    float* output_buffer,
    int* output_counts,
    int num_neurons,
    int num_outputs,
    int group_size);

// ============================================================================
// MODULAR ARCHITECTURE SUPPORT KERNELS
// ============================================================================

/**
 * @brief Initialize modular neural network with self-contained modules
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments per neuron
 * @param module_configs Device pointer to per-module configuration
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 * @param num_modules Number of independent modules
 */
void initialize_modular_network(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int* module_assignments,
    void* module_configs,  // ModuleConfiguration array
    int num_neurons,
    int num_synapses,
    int num_modules
);

/**
 * @brief Update modules with independent processing and state management
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments
 * @param module_states Device pointer to per-module state data
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 * @param num_modules Number of modules
 */
void update_modular_states(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int* module_assignments,
    void* module_states,
    float current_time,
    float dt,
    int num_neurons,
    int num_synapses,
    int num_modules
);

/**
 * @brief Process central attention/control mechanism for module orchestration
 * @param neurons Device pointer to neuron array
 * @param module_assignments Device pointer to module ID assignments
 * @param attention_weights Device pointer to per-module attention weights
 * @param context_input Device pointer to current context/input information
 * @param global_inhibition Device pointer to inter-module inhibition levels
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_modules Number of modules
 */
void process_central_attention(
    GPUNeuronState* neurons,
    int* module_assignments,
    float* attention_weights,
    float* context_input,
    float* global_inhibition,
    float current_time,
    int num_neurons,
    int num_modules
);

/**
 * @brief Manage inter-modular communication and feedback loops
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments
 * @param inter_module_matrix Device pointer to module-to-module connectivity
 * @param feedback_states Device pointer to internal feedback state
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 * @param num_modules Number of modules
 */
void process_inter_modular_communication(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int* module_assignments,
    float* inter_module_matrix,
    void* feedback_states,
    float current_time,
    int num_neurons,
    int num_synapses,
    int num_modules
);

/**
 * @brief Save state of specific module for independent persistence
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments
 * @param module_state_buffer Device pointer to output buffer for module state
 * @param module_id ID of the module to save
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void save_module_state(
    const GPUNeuronState* neurons,
    const GPUSynapse* synapses,
    const int* module_assignments,
    void* module_state_buffer,
    int module_id,
    int num_neurons,
    int num_synapses
);

/**
 * @brief Load state of specific module for independent restoration
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments
 * @param module_state_buffer Device pointer to input buffer with module state
 * @param module_id ID of the module to restore
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void load_module_state(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    const int* module_assignments,
    const void* module_state_buffer,
    int module_id,
    int num_neurons,
    int num_synapses
);

// ============================================================================
// SPECIALIZED MODULE PROCESSING
// ============================================================================

/**
 * @brief Process specialized sensory input modules
 * @param neurons Device pointer to neuron array
 * @param module_assignments Device pointer to module ID assignments
 * @param sensory_inputs Device pointer to multi-modal sensory data
 * @param input_routing Device pointer to routing matrix for different inputs
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_sensory_modules Number of sensory processing modules
 */
void process_sensory_modules(
    GPUNeuronState* neurons,
    int* module_assignments,
    float* sensory_inputs,
    int* input_routing,
    float current_time,
    int num_neurons,
    int num_sensory_modules
);

/**
 * @brief Process memory and association modules with context-dependent activation
 * @param neurons Device pointer to neuron array
 * @param synapses Device pointer to synapse array
 * @param module_assignments Device pointer to module ID assignments
 * @param memory_traces Device pointer to long-term memory traces
 * @param context_signals Device pointer to current context for memory retrieval
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 * @param num_memory_modules Number of memory processing modules
 */
void process_memory_modules(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int* module_assignments,
    void* memory_traces,
    float* context_signals,
    float current_time,
    int num_neurons,
    int num_synapses,
    int num_memory_modules
);

/**
 * @brief Process executive control modules for high-level coordination
 * @param neurons Device pointer to neuron array
 * @param module_assignments Device pointer to module ID assignments
 * @param goal_states Device pointer to current goal/task representations
 * @param control_signals Device pointer to output control signals
 * @param decision_history Device pointer to previous decision history
 * @param current_time Current simulation time (ms)
 * @param num_neurons Total number of neurons
 * @param num_control_modules Number of executive control modules
 */
void process_executive_modules(
    GPUNeuronState* neurons,
    int* module_assignments,
    float* goal_states,
    float* control_signals,
    void* decision_history,
    float current_time,
    int num_neurons,
    int num_control_modules
);

// ============================================================================
// PERFORMANCE AND DEBUGGING UTILITIES
// ============================================================================

/**
 * @brief Collect comprehensive statistics from all modules
 * @param neurons Device pointer to neuron array (read-only)
 * @param synapses Device pointer to synapse array (read-only)
 * @param module_assignments Device pointer to module ID assignments
 * @param stats_buffer Device pointer to output statistics buffer
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 * @param num_modules Number of modules
 */
void collect_modular_statistics(
    const GPUNeuronState* neurons,
    const GPUSynapse* synapses,
    const int* module_assignments,
    void* stats_buffer,
    int num_neurons,
    int num_synapses,
    int num_modules
);

/**
 * @brief Check network health and detect potential issues
 * @param neurons Device pointer to neuron array (read-only)
 * @param synapses Device pointer to synapse array (read-only)
 * @param health_report Device pointer to output health report
 * @param num_neurons Total number of neurons
 * @param num_synapses Total number of synapses
 */
void check_network_health(
    const GPUNeuronState* neurons,
    const GPUSynapse* synapses,
    void* health_report,
    int num_neurons,
    int num_synapses
);

} // namespace KernelLaunchWrappers

// ============================================================================
// MODULAR ARCHITECTURE CONFIGURATION STRUCTURES
// ============================================================================

/**
 * @brief Configuration for individual neural modules
 */
struct ModuleConfiguration {
    int module_id;                    // Unique identifier
    int start_neuron_idx;            // Starting neuron index in global array
    int end_neuron_idx;              // Ending neuron index (exclusive)
    int start_synapse_idx;           // Starting synapse index in global array
    int end_synapse_idx;             // Ending synapse index (exclusive)
    
    // Module-specific parameters
    float learning_rate_modifier;    // Module-specific learning rate scaling
    float attention_sensitivity;     // How much this module responds to attention
    float inter_module_coupling;     // Strength of coupling to other modules
    
    // Specialized function flags
    bool is_sensory_module;          // Processes sensory input
    bool is_memory_module;           // Handles memory/association
    bool is_executive_module;        // Executive control functions
    bool is_output_module;           // Generates network output
    
    // Dynamic reconfiguration
    bool can_resize;                 // Whether this module can change size
    int min_neurons;                 // Minimum neurons if resizable
    int max_neurons;                 // Maximum neurons if resizable
    
    // State management
    void* persistent_state;          // Pointer to module-specific persistent state
    size_t state_size;              // Size of persistent state in bytes
};

/**
 * @brief Central attention and control state
 */
struct AttentionState {
    float attention_weights[256];    // Per-module attention weights (0-1)
    float global_inhibition[256];    // Per-module global inhibition levels
    float context_vector[64];        // Current context representation
    float goal_vector[32];           // Current goal/task representation
    
    // Attention dynamics
    float attention_decay_rate;     // How quickly attention decays
    float attention_update_rate;    // How quickly attention adapts
    float inhibition_strength;      // Strength of inter-module competition
    
    // Context processing
    int active_context_modules[16]; // Currently active context-processing modules
    int num_active_context;         // Number of active context modules
    
    // Executive control
    int current_task_id;            // Current task/goal identifier
    float task_urgency;             // Urgency of current task (0-1)
    float decision_confidence;      // Confidence in current processing state
};

#endif // KERNEL_LAUNCH_WRAPPERS_CUH