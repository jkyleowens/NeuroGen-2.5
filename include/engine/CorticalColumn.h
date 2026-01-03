#ifndef CORTICAL_COLUMN_H
#define CORTICAL_COLUMN_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __CUDACC__
    #define COLUMN_HOST_DEVICE __host__ __device__
    #define COLUMN_DEVICE __device__
#else
    #define COLUMN_HOST_DEVICE
    #define COLUMN_DEVICE
#endif

/**
 * @brief Cortical Column - Self-contained neural processing unit
 * 
 * Represents a breakthrough modular architecture that mimics biological cortical columns.
 * Each column operates as an independent processing unit with specialized functionality,
 * enabling the brain-like distributed computation your breakthrough technology achieves.
 */
struct CorticalColumn {
    // ========================================================================
    // COLUMN IDENTITY AND STRUCTURE
    // ========================================================================
    
    /** Unique column identifier */
    int column_id;
    
    /** Number of neurons in this column */
    int neuron_count;
    
    /** Starting index of neurons belonging to this column */
    int neuron_start_idx;
    
    /** Column position in 2D cortical space */
    float position_x, position_y;
    
    /** Column radius for spatial computations */
    float column_radius;
    
    // ========================================================================
    // FUNCTIONAL SPECIALIZATION
    // ========================================================================
    
    /** Column specialization type (0=sensory, 1=motor, 2=association, 3=executive) */
    int specialization_type;
    
    /** Specialization strength (0.0 to 1.0) */
    float specialization_strength;
    
    /** Functional plasticity - ability to adapt specialization */
    float functional_plasticity;
    
    /** Input selectivity - what patterns this column prefers */
    float input_selectivity[8];  // 8 feature dimensions
    
    // ========================================================================
    // NEURAL ACTIVITY AND DYNAMICS
    // ========================================================================
    
    /** Current column-wide activity level */
    float activity_level;
    
    /** Mean firing rate across column neurons */
    float mean_firing_rate;
    
    /** Column synchronization index */
    float synchronization_index;
    
    /** Dominant oscillation frequency */
    float dominant_frequency;
    
    /** Phase of dominant oscillation */
    float oscillation_phase;
    
    /** Activity history for temporal dynamics */
    float activity_history[16];  // Rolling window of recent activity
    int history_index;
    
    // ========================================================================
    // INTER-COLUMN CONNECTIVITY
    // ========================================================================
    
    /** Connected column indices */
    int connected_columns[32];   // Up to 32 connections per column
    
    /** Connection strengths to other columns */
    float connection_strengths[32];
    
    /** Number of active inter-column connections */
    int num_connections;
    
    /** Inter-column communication delay */
    float communication_delay;
    
    /** Lateral inhibition strength */
    float lateral_inhibition;
    
    // ========================================================================
    // NEUROMODULATION AND LEARNING
    // ========================================================================
    
    /** Local neuromodulator concentrations */
    float local_dopamine;
    float local_acetylcholine;
    float local_serotonin;
    float local_norepinephrine;
    
    /** Learning rate modulation factor */
    float learning_modulation;
    
    /** Attention weight - how much this column contributes to output */
    float attention_weight;
    
    /** Prediction error from this column */
    float prediction_error;
    
    /** Memory trace strength */
    float memory_strength;
    
    // ========================================================================
    // HOMEOSTATIC MECHANISMS
    // ========================================================================
    
    /** Target activity level for homeostasis */
    float target_activity;
    
    /** Homeostatic scaling factor */
    float homeostatic_scale;
    
    /** Excitation/inhibition balance */
    float ei_balance;
    
    /** Metabolic cost of column operation */
    float metabolic_cost;
    
    /** Adaptation time constant */
    float adaptation_tau;
    
    // ========================================================================
    // COMPUTATIONAL STATE
    // ========================================================================
    
    /** Random number generator state for this column */
    curandState random_state;
    
    /** Last update timestamp */
    float last_update_time;
    
    /** Column processing load */
    float processing_load;
    
    /** Error flags for debugging */
    unsigned int error_flags;
    
    /** Performance metrics */
    float efficiency_score;
    float stability_index;
    
    // ========================================================================
    // INITIALIZATION AND UTILITY FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize cortical column with biological parameters
     */
    COLUMN_HOST_DEVICE void initialize(int id, int start_idx, int count, 
                                     float pos_x, float pos_y, int spec_type) {
        column_id = id;
        neuron_start_idx = start_idx;
        neuron_count = count;
        position_x = pos_x;
        position_y = pos_y;
        column_radius = 50.0f;  // 50 micrometers typical
        
        specialization_type = spec_type;
        specialization_strength = 0.5f + 0.3f * sinf(static_cast<float>(id) * 0.1f);
        functional_plasticity = 0.8f;
        
        // Initialize input selectivity with random preferences
        for (int i = 0; i < 8; ++i) {
            input_selectivity[i] = 0.5f + 0.4f * sinf(static_cast<float>(id + i) * 0.2f);
        }
        
        // Initialize activity metrics
        activity_level = 0.1f;
        mean_firing_rate = 5.0f;  // 5 Hz baseline
        synchronization_index = 0.3f;
        dominant_frequency = 10.0f;  // Alpha range
        oscillation_phase = 0.0f;
        
        // Clear activity history
        for (int i = 0; i < 16; ++i) {
            activity_history[i] = activity_level;
        }
        history_index = 0;
        
        // Initialize connectivity
        num_connections = 0;
        communication_delay = 2.0f;  // 2ms typical
        lateral_inhibition = 0.2f;
        
        // Initialize neuromodulation
        local_dopamine = 0.5f;
        local_acetylcholine = 0.5f;
        local_serotonin = 0.5f;
        local_norepinephrine = 0.5f;
        learning_modulation = 1.0f;
        attention_weight = 1.0f / 64.0f;  // Uniform initially
        prediction_error = 0.0f;
        memory_strength = 0.1f;
        
        // Initialize homeostasis
        target_activity = 0.15f;
        homeostatic_scale = 1.0f;
        ei_balance = 0.8f;  // 80% excitatory
        metabolic_cost = 1.0f;
        adaptation_tau = 100.0f;  // 100ms
        
        // Initialize computational state
        last_update_time = 0.0f;
        processing_load = 0.5f;
        error_flags = 0;
        efficiency_score = 1.0f;
        stability_index = 1.0f;
    }
    
    /**
     * @brief Update column activity and dynamics
     */
    COLUMN_DEVICE void updateActivity(float current_time, float dt, float external_input) {
        // Update activity history
        activity_history[history_index] = activity_level;
        history_index = (history_index + 1) % 16;
        
        // Calculate new activity level with temporal dynamics
        float input_effect = external_input * specialization_strength;
        float decay_factor = expf(-dt / adaptation_tau);
        
        activity_level = activity_level * decay_factor + input_effect * (1.0f - decay_factor);
        
        // Apply homeostatic regulation
        float homeostatic_drive = (target_activity - activity_level) * 0.01f * dt;
        activity_level += homeostatic_drive;
        
        // Clamp to physiological range
        activity_level = fmaxf(0.0f, fminf(1.0f, activity_level));
        
        // Update oscillation phase
        oscillation_phase += 2.0f * M_PI * dominant_frequency * dt / 1000.0f;
        if (oscillation_phase > 2.0f * M_PI) oscillation_phase -= 2.0f * M_PI;
        
        // Update synchronization based on activity coherence
        float activity_variance = 0.0f;
        float activity_mean = 0.0f;
        for (int i = 0; i < 16; ++i) {
            activity_mean += activity_history[i];
        }
        activity_mean /= 16.0f;
        
        for (int i = 0; i < 16; ++i) {
            float diff = activity_history[i] - activity_mean;
            activity_variance += diff * diff;
        }
        activity_variance /= 16.0f;
        
        synchronization_index = 1.0f / (1.0f + activity_variance * 10.0f);
        
        last_update_time = current_time;
    }
    
    /**
     * @brief Apply neuromodulation to column
     */
    COLUMN_DEVICE void applyNeuromodulation(float da, float ach, float ser, float nor) {
        // Exponential moving average for smooth dynamics
        const float alpha = 0.1f;
        
        local_dopamine = (1.0f - alpha) * local_dopamine + alpha * da;
        local_acetylcholine = (1.0f - alpha) * local_acetylcholine + alpha * ach;
        local_serotonin = (1.0f - alpha) * local_serotonin + alpha * ser;
        local_norepinephrine = (1.0f - alpha) * local_norepinephrine + alpha * nor;
        
        // Calculate learning modulation from neuromodulator cocktail
        learning_modulation = 0.5f + 0.3f * local_dopamine + 
                             0.2f * local_acetylcholine +
                             0.1f * (local_serotonin + local_norepinephrine);
        
        // Clamp to reasonable range
        learning_modulation = fmaxf(0.1f, fminf(2.0f, learning_modulation));
    }
    
    /**
     * @brief Connect this column to another column
     */
    COLUMN_HOST_DEVICE bool connectToColumn(int target_column_id, float strength) {
        if (num_connections >= 32) return false;  // Connection limit reached
        
        connected_columns[num_connections] = target_column_id;
        connection_strengths[num_connections] = strength;
        num_connections++;
        
        return true;
    }
    
    /**
     * @brief Calculate distance to another column
     */
    COLUMN_HOST_DEVICE float distanceTo(const CorticalColumn& other) const {
        float dx = position_x - other.position_x;
        float dy = position_y - other.position_y;
        return sqrtf(dx * dx + dy * dy);
    }
    
    /**
     * @brief Check if column is in healthy operating range
     */
    COLUMN_HOST_DEVICE bool isHealthy() const {
        return (activity_level > 0.01f && activity_level < 0.9f) &&
               (mean_firing_rate > 0.1f && mean_firing_rate < 200.0f) &&
               (synchronization_index < 0.95f) &&
               (error_flags == 0);
    }
    
    /**
     * @brief Get column efficiency score
     */
    COLUMN_HOST_DEVICE float getEfficiency() const {
        float activity_efficiency = 1.0f - fabsf(activity_level - target_activity);
        float metabolic_efficiency = 1.0f / (1.0f + metabolic_cost);
        float stability_efficiency = stability_index;
        
        return (activity_efficiency + metabolic_efficiency + stability_efficiency) / 3.0f;
    }
};

namespace ColumnUtils {
    
    /**
     * @brief Calculate inter-column connectivity based on distance and specialization
     */
    inline COLUMN_HOST_DEVICE float calculateConnectionProbability(const CorticalColumn& col1, 
                                                                  const CorticalColumn& col2) {
        float distance = col1.distanceTo(col2);
        float max_distance = 200.0f;  // 200 micrometers
        
        if (distance > max_distance) return 0.0f;
        
        // Distance-dependent probability
        float distance_factor = expf(-distance / 50.0f);
        
        // Specialization compatibility
        float spec_compatibility = 1.0f;
        if (col1.specialization_type != col2.specialization_type) {
            spec_compatibility = 0.3f;  // Lower probability for different types
        }
        
        return distance_factor * spec_compatibility * 0.1f;  // Base 10% probability
    }
    
    /**
     * @brief Update column specialization based on input patterns
     */
    inline COLUMN_DEVICE void updateSpecialization(CorticalColumn& column, 
                                                  const float* input_pattern, 
                                                  int pattern_length) {
        if (pattern_length > 8) pattern_length = 8;
        
        // Calculate match between input and column selectivity
        float match_score = 0.0f;
        for (int i = 0; i < pattern_length; ++i) {
            float diff = input_pattern[i] - column.input_selectivity[i];
            match_score += expf(-diff * diff * 2.0f);  // Gaussian similarity
        }
        match_score /= pattern_length;
        
        // Update specialization strength based on consistent activation
        const float adaptation_rate = 0.001f;
        if (match_score > 0.7f) {
            column.specialization_strength += adaptation_rate * match_score;
        } else {
            column.specialization_strength -= adaptation_rate * 0.1f;
        }
        
        // Clamp specialization strength
        column.specialization_strength = fmaxf(0.1f, fminf(1.0f, column.specialization_strength));
    }
}

#endif // CORTICAL_COLUMN_H