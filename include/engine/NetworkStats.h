#ifndef NETWORK_STATS_H
#define NETWORK_STATS_H

#include <cstdint>

/**
 * @brief Network statistics structure
 * 
 * Tracks various metrics about neural network performance
 */
struct NetworkStats {
    // Activity metrics
    float average_firing_rate = 0.0f;
    float average_membrane_potential = 0.0f;
    int active_neurons = 0;
    int total_spikes = 0;
    
    // Synaptic metrics
    float average_synaptic_weight = 0.0f;
    float total_synaptic_current = 0.0f;
    
    // Learning metrics
    float total_weight_change = 0.0f;
    float average_eligibility_trace = 0.0f;
    
    // Performance metrics
    float simulation_time_ms = 0.0f;
    float computation_time_ms = 0.0f;
    uint64_t update_count = 0;
    
    // Network health
    bool is_stable = true;
    float stability_measure = 1.0f;
    
    // Memory usage
    size_t memory_usage_bytes = 0;
};

#endif // NETWORK_STATS_H

