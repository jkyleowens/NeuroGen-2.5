#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <cstddef>

/**
 * @brief Configuration for neural network structure
 */
struct NetworkConfig {
    // Network size
    size_t num_neurons = 1000;
    size_t num_synapses = 10000;
    size_t num_inputs = 100;
    size_t num_outputs = 100;
    
    // Network topology
    float percent_inhibitory = 0.2f;
    float connection_density = 0.1f;
    
    // Neuron parameters
    float excitatory_weight_range = 1.0f;
    float inhibitory_weight_range = 1.0f;
    
    // Learning parameters
    float learning_rate = 0.01f;
    bool enable_plasticity = true;
    bool enable_stdp = true;
    bool enable_homeostasis = true;
    
    // Simulation parameters
    float time_step_ms = 1.0f;
    float simulation_duration_ms = 1000.0f;
};

#endif // NETWORK_CONFIG_H

