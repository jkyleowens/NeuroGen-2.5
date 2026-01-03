#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include <engine/NeuralConstants.h>

// ============================================================================
// COMPLETE GPU PLASTICITY STATE STRUCTURE - MISSING DEFINITION
// ============================================================================

struct GPUPlasticityState {
    // === CORE PLASTICITY PARAMETERS ===
    float stdp_learning_rate;               // STDP learning rate
    float bcm_learning_rate;                // BCM learning rate
    float homeostatic_rate;                 // Homeostatic scaling rate
    float metaplasticity_rate;              // Meta-plasticity modulation rate
    
    // === TIMING AND THRESHOLDS ===
    float stdp_window_ltp;                  // LTP time window (ms)
    float stdp_window_ltd;                  // LTD time window (ms)
    float bcm_threshold;                    // BCM sliding threshold
    float plasticity_threshold;             // General plasticity threshold
    
    // === ACTIVITY TRACKING ===
    float total_weight_change;              // Cumulative weight changes
    float plasticity_events_count;          // Number of plasticity events
    float average_eligibility_trace;        // Mean eligibility trace
    float last_update_time;                 // Last plasticity update time
    
    // === CALCIUM AND PROTEIN SYNTHESIS ===
    float calcium_threshold_ltp;            // Calcium threshold for LTP
    float calcium_threshold_ltd;            // Calcium threshold for LTD
    float protein_synthesis_rate;           // Rate of protein synthesis
    float late_phase_threshold;             // Threshold for late-phase plasticity
    
    // === ELIGIBILITY TRACES ===
    float trace_decay_rate;                 // Eligibility trace decay rate
    float trace_amplitude;                  // Maximum trace amplitude
    float trace_integration_window;         // Integration time window
    
    // === META-PLASTICITY ===
    float recent_activity_level;            // Recent synaptic activity
    float metaplasticity_threshold;         // Meta-plasticity activation threshold
    float sliding_threshold_rate;           // BCM threshold adaptation rate
    
    // === CONSTRAINTS AND BOUNDS ===
    float min_weight_change;                // Minimum detectable weight change
    float max_weight_change;                // Maximum single-step weight change
    float saturation_factor;                // Weight saturation factor
    
    // === STATE FLAGS ===
    bool stdp_enabled;                      // STDP mechanism enabled
    bool bcm_enabled;                       // BCM mechanism enabled
    bool homeostatic_enabled;               // Homeostatic scaling enabled
    bool metaplasticity_enabled;            // Meta-plasticity enabled
    bool late_phase_plasticity_enabled;     // Late-phase LTP/LTD enabled
};

// ============================================================================
// COMPLETE GPU NEUROMODULATOR STATE STRUCTURE - MISSING DEFINITION
// ============================================================================

struct GPUNeuromodulatorState {
    // === PRIMARY NEUROMODULATORS ===
    float dopamine_concentration;           // Global dopamine level
    float acetylcholine_concentration;      // Global acetylcholine level
    float serotonin_concentration;          // Global serotonin level
    float norepinephrine_concentration;     // Global norepinephrine level
    float gaba_concentration;               // Global GABA level
    float glutamate_concentration;          // Global glutamate level
    
    // === RECEPTOR DENSITIES ===
    float dopamine_d1_density;              // D1 receptor density
    float dopamine_d2_density;              // D2 receptor density
    float acetylcholine_nic_density;        // Nicotinic receptor density
    float acetylcholine_musc_density;       // Muscarinic receptor density
    float serotonin_5ht1a_density;          // 5-HT1A receptor density
    float serotonin_5ht2a_density;          // 5-HT2A receptor density
    
    // === RELEASE AND UPTAKE DYNAMICS ===
    float dopamine_release_rate;            // Dopamine release rate
    float dopamine_uptake_rate;             // Dopamine reuptake rate
    float acetylcholine_release_rate;       // ACh release rate
    float acetylcholine_degradation_rate;   // ACh degradation rate
    float serotonin_release_rate;           // Serotonin release rate
    float serotonin_uptake_rate;            // Serotonin reuptake rate
    
    // === MODULATION PARAMETERS ===
    float learning_modulation_strength;     // Learning rate modulation
    float attention_modulation_strength;    // Attention gating strength
    float memory_consolidation_strength;    // Memory consolidation modulation
    float plasticity_gating_threshold;      // Plasticity gating threshold
    
    // === TIMING AND DYNAMICS ===
    float modulation_time_constant;         // Modulation dynamics time constant
    float baseline_recovery_rate;           // Return to baseline rate
    float peak_modulation_duration;         // Duration of peak modulation
    float last_modulation_time;             // Time of last significant modulation
    
    // === SPATIAL DISTRIBUTION ===
    float cortical_modulation_level;        // Cortical modulation strength
    float hippocampal_modulation_level;     // Hippocampal modulation strength
    float striatal_modulation_level;        // Striatal modulation strength
    float thalamic_modulation_level;        // Thalamic modulation strength
    
    // === REWARD AND PREDICTION ===
    float reward_prediction_error;          // Current RPE signal
    float expected_reward;                  // Expected reward value
    float actual_reward;                    // Actual received reward
    float reward_sensitivity;               // Sensitivity to reward signals
    
    // === HOMEOSTATIC REGULATION ===
    float baseline_dopamine;                // Baseline dopamine level
    float baseline_acetylcholine;           // Baseline acetylcholine level
    float baseline_serotonin;               // Baseline serotonin level
    float homeostatic_recovery_rate;        // Rate of return to baseline
    
    // === STATE FLAGS ===
    bool dopamine_system_active;            // Dopamine system active
    bool acetylcholine_system_active;       // Acetylcholine system active
    bool serotonin_system_active;           // Serotonin system active
    bool modulation_enabled;                // Overall modulation enabled
    bool reward_prediction_enabled;         // Reward prediction active
};

// ============================================================================
// EXISTING STRUCTURES (keeping for completeness)
// ============================================================================

struct GPUNeuronState {
    // === CORE MEMBRANE DYNAMICS ===
    float V;                            // Membrane potential (mV)
    float u;                            // Recovery variable (Izhikevich)
    float I_syn[MAX_COMPARTMENTS];      // Synaptic currents per compartment
    float I_ext;                        // External current input
    float ca_conc[MAX_COMPARTMENTS];    // Calcium concentrations
    
    // === TIMING AND ACTIVITY ===
    float last_spike_time;              // Time of last spike
    float previous_spike_time;          // Previous spike time
    float average_firing_rate;          // Running average firing rate
    float average_activity;             // Average activity level
    float activity_level;               // Current activity level
    
    // === PLASTICITY AND ADAPTATION ===
    float excitability;                 // Intrinsic excitability
    float synaptic_scaling_factor;      // Global synaptic scaling
    float bcm_threshold;                // BCM learning threshold
    float plasticity_threshold;         // Plasticity induction threshold
    float threshold;                    // Firing threshold
    float firing_rate;                  // Instantaneous firing rate
    
    // === NEUROMODULATION ===
    float dopamine_concentration;       // Local dopamine level
    float acetylcholine_level;          // Local acetylcholine level
    float serotonin_level;              // Local serotonin level
    float norepinephrine_level;         // Local norepinephrine level
    
    // === ION CHANNELS ===
    float na_m, na_h;                   // Sodium channel states
    float k_n;                          // Potassium channel state
    float ca_channel_state;             // Calcium channel state
    float channel_expression[NUM_RECEPTOR_TYPES]; // Channel expression levels
    float channel_maturation[NUM_RECEPTOR_TYPES]; // Channel maturation states
    
    // === MULTI-COMPARTMENT SUPPORT ===
    float V_compartments[MAX_COMPARTMENTS];        // Compartment voltages
    int compartment_types[MAX_COMPARTMENTS];       // Compartment types
    int num_compartments;                          // Number of active compartments
    bool dendritic_spike[MAX_DENDRITIC_SPIKES];    // Dendritic spike states
    float dendritic_spike_time[MAX_DENDRITIC_SPIKES]; // Dendritic spike timing
    
    // === NETWORK PROPERTIES ===
    int neuron_type;                    // Neuron type (excitatory/inhibitory)
    int layer_id;                       // Cortical layer
    int column_id;                      // Cortical column
    int active;                         // Activity flag
    bool is_principal_cell;             // Principal vs interneuron
    
    // === SPATIAL PROPERTIES ===
    float position_x, position_y, position_z;     // 3D coordinates
    float orientation_theta;            // Orientation
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Current development stage
    float maturation_factor;            // Maturation level [0,1]
    float birth_time;                   // Time of neurogenesis
    
    // === METABOLISM ===
    float energy_level;                 // Cellular energy
    float metabolic_demand;             // Energy demand
    float glucose_uptake;               // Glucose consumption rate
};

struct GPUSynapse {
    // === CONNECTIVITY ===
    int pre_neuron_idx;                 // Presynaptic neuron index
    int post_neuron_idx;                // Postsynaptic neuron index
    int target_neuron_idx;              // Target neuron index (for homeostatic scaling)
    int post_compartment;               // Target compartment
    int receptor_index;                 // Receptor type
    int active;                         // Activity flag
    
    // === SYNAPTIC PROPERTIES ===
    float weight;                       // Current weight
    float max_weight, min_weight;       // Weight bounds
    float delay;                        // Synaptic delay
    float effective_weight;             // Modulated weight
    
    // === PLASTICITY ===
    float eligibility_trace;            // Eligibility trace
    float plasticity_modulation;        // Plasticity modulation
    bool is_plastic;                    // Plasticity enabled flag
    float learning_rate;                // Synapse-specific learning rate
    float metaplasticity_factor;        // Meta-plasticity scaling
    
    // === TIMING ===
    float last_pre_spike_time;          // Last presynaptic spike
    float last_post_spike_time;         // Last postsynaptic spike
    float last_active_time;             // Last activation time
    float activity_metric;              // Activity measure
    float last_potentiation;            // Last potentiation time
    
    // === NEUROMODULATION ===
    float dopamine_sensitivity;         // Dopamine sensitivity
    float acetylcholine_sensitivity;    // ACh sensitivity
    float serotonin_sensitivity;        // Serotonin sensitivity
    float dopamine_level;               // Local dopamine
    
    // === VESICLE DYNAMICS ===
    int vesicle_count;                  // Available vesicles
    float release_probability;          // Release probability
    float facilitation_factor;          // Short-term facilitation
    float depression_factor;            // Short-term depression
    
    // === CALCIUM DYNAMICS ===
    float presynaptic_calcium;          // Pre-synaptic calcium
    float postsynaptic_calcium;         // Post-synaptic calcium
    
    // === HOMEOSTASIS ===
    float homeostatic_scaling;          // Homeostatic scaling
    float target_activity;              // Target activity level
    
    // === BIOPHYSICS ===
    float conductance;                  // Synaptic conductance
    float reversal_potential;           // Reversal potential
    float time_constant_rise;           // Rise time constant
    float time_constant_decay;          // Decay time constant
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Development stage
    float structural_stability;         // Resistance to pruning
    float growth_factor;                // Growth tendency
};

// ============================================================================
// STRUCTURE OF ARRAYS (SoA) - OPTIMIZED FOR GPU COALESCING
// ============================================================================

/**
 * @brief Structure of Arrays layout for neurons (optimized for GPU coalescing)
 * 
 * Benefits:
 * - 2-3x memory bandwidth improvement from coalesced access
 * - Better cache utilization
 * - Reduced register pressure in kernels
 */
struct NeuronArrays {
    // === CORE MEMBRANE DYNAMICS (hot path) ===
    float* V;                           // Membrane potential (mV)
    float* u;                           // Recovery variable (Izhikevich)
    float* I_syn_0;                     // Synaptic current compartment 0
    float* I_syn_1;                     // Synaptic current compartment 1
    float* I_syn_2;                     // Synaptic current compartment 2
    float* I_syn_3;                     // Synaptic current compartment 3
    float* I_ext;                       // External current input
    
    // === CALCIUM DYNAMICS (hot path) ===
    float* ca_conc_0;                   // Calcium concentration compartment 0
    float* ca_conc_1;                   // Calcium concentration compartment 1
    float* ca_conc_2;                   // Calcium concentration compartment 2
    float* ca_conc_3;                   // Calcium concentration compartment 3
    
    // === TIMING (hot path) ===
    float* last_spike_time;             // Time of last spike
    float* previous_spike_time;         // Previous spike time
    
    // === ACTIVITY (medium frequency) ===
    float* average_firing_rate;         // Running average firing rate
    float* average_activity;            // Average activity level
    float* activity_level;              // Current activity level
    float* firing_rate;                 // Instantaneous firing rate
    
    // === PLASTICITY (medium frequency) ===
    float* excitability;                // Intrinsic excitability
    float* synaptic_scaling_factor;     // Global synaptic scaling
    float* bcm_threshold;               // BCM learning threshold
    float* plasticity_threshold;        // Plasticity induction threshold
    float* threshold;                   // Firing threshold
    
    // === NEUROMODULATION (medium frequency) ===
    float* dopamine_concentration;      // Local dopamine level
    float* acetylcholine_level;         // Local acetylcholine level
    float* serotonin_level;             // Local serotonin level
    float* norepinephrine_level;        // Local norepinephrine level
    
    // === ION CHANNELS (cold path) ===
    float* na_m;                        // Sodium channel activation
    float* na_h;                        // Sodium channel inactivation
    float* k_n;                         // Potassium channel state
    float* ca_channel_state;            // Calcium channel state
    
    // === NETWORK PROPERTIES (cold path) ===
    int* neuron_type;                   // Neuron type (excitatory/inhibitory)
    int* layer_id;                      // Cortical layer
    int* column_id;                     // Cortical column
    int* active;                        // Activity flag
    
    // === METABOLISM (cold path) ===
    float* energy_level;                // Cellular energy
    float* metabolic_demand;            // Energy demand
    
    size_t num_neurons;                 // Total number of neurons
};

/**
 * @brief Structure of Arrays layout for synapses (optimized for GPU coalescing)
 */
struct SynapseArrays {
    // === CONNECTIVITY (hot path) ===
    int* pre_neuron_idx;                // Presynaptic neuron index
    int* post_neuron_idx;               // Postsynaptic neuron index
    int* post_compartment;              // Target compartment
    int* active;                        // Activity flag
    
    // === SYNAPTIC PROPERTIES (hot path) ===
    float* weight;                      // Current weight
    float* max_weight;                  // Maximum weight bound
    float* min_weight;                  // Minimum weight bound
    float* effective_weight;            // Modulated weight
    
    // === PLASTICITY (hot path) ===
    float* eligibility_trace;           // Eligibility trace
    float* learning_rate;               // Synapse-specific learning rate
    
    // === TIMING (hot path) ===
    float* last_pre_spike_time;         // Last presynaptic spike
    float* last_post_spike_time;        // Last postsynaptic spike
    
    // === NEUROMODULATION (medium frequency) ===
    float* dopamine_sensitivity;        // Dopamine sensitivity
    float* dopamine_level;              // Local dopamine
    
    // === SHORT-TERM PLASTICITY (medium frequency) ===
    float* release_probability;         // Release probability
    float* facilitation_factor;         // Short-term facilitation
    float* depression_factor;           // Short-term depression
    
    // === CALCIUM DYNAMICS (medium frequency) ===
    float* presynaptic_calcium;         // Pre-synaptic calcium
    float* postsynaptic_calcium;        // Post-synaptic calcium
    
    // === OTHER PROPERTIES (cold path) ===
    float* plasticity_modulation;       // Plasticity modulation
    float* metaplasticity_factor;       // Meta-plasticity scaling
    float* delay;                       // Synaptic delay
    float* homeostatic_scaling;         // Homeostatic scaling
    int* receptor_index;                // Receptor type
    int* vesicle_count;                 // Available vesicles
    
    size_t num_synapses;                // Total number of synapses
};

// ============================================================================
// GPU NETWORK CONFIGURATION STRUCTURE
// ============================================================================

struct GPUNetworkConfig {
    // === LEARNING PARAMETERS ===
    float global_learning_rate;         // Overall learning rate
    float stdp_learning_rate;           // STDP-specific rate
    float bcm_learning_rate;            // BCM-specific rate
    float homeostatic_rate;             // Homeostatic scaling rate
    
    // === TIMING PARAMETERS ===
    float simulation_dt;                // Simulation time step
    float plasticity_update_interval;   // Plasticity update frequency
    float monitoring_interval;          // Monitoring frequency
    
    // === NETWORK DIMENSIONS ===
    int num_neurons;                    // Total neuron count
    int num_synapses;                   // Total synapse count
    int num_modules;                    // Number of neural modules
    
    // === ACTIVATION PARAMETERS ===
    float noise_amplitude;              // Background noise level
    float input_scaling;                // Input signal scaling
    float output_scaling;               // Output signal scaling
    
    // === STABILITY PARAMETERS ===
    float max_weight_change;            // Maximum weight change per step
    float stability_threshold;          // Network stability threshold
    float convergence_criterion;        // Learning convergence criterion
};

#endif // GPU_NEURAL_STRUCTURES_H