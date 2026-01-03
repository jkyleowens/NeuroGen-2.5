#ifndef LEARNING_STATE_H
#define LEARNING_STATE_H

#include <vector>
#include <cstdint>

/**
 * @brief Learning state management
 * 
 * Tracks learning progress, eligibility traces, and consolidation state
 */
struct LearningState {
    // Eligibility traces
    std::vector<float> eligibility_traces;
    std::vector<float> synaptic_tags;
    
    // Neuromodulator levels
    float dopamine_level = 0.0f;
    float serotonin_level = 0.0f;
    float acetylcholine_level = 0.0f;
    
    // Learning statistics
    float total_weight_change = 0.0f;
    float average_firing_rate = 0.0f;
    uint64_t learning_steps = 0;
    
    // Consolidation state
    bool consolidation_active = false;
    float consolidation_progress = 0.0f;
    
    // Module assignments
    std::vector<int> module_assignments;
    int num_modules = 0;
};

#endif // LEARNING_STATE_H

