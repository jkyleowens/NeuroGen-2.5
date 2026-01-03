#pragma once

namespace neurogen {

/**
 * @brief Learning mode for different brain regions
 * 
 * Implements the "split-brain" learning strategy where sensory/memory
 * regions use unsupervised STDP while action/control regions use
 * reward-modulated learning.
 */
enum class LearningMode {
    /**
     * Pure Spike-Timing Dependent Plasticity (Unsupervised)
     * - Uses only pre/post spike correlations
     * - Reward signal is IGNORED
     * - Suitable for: sensory processing, feature extraction, memory
     * - Biological analog: early sensory cortex, hippocampus
     */
    PURE_STDP,
    
    /**
     * Reward-Modulated STDP (Three-Factor Rule)
     * - Uses eligibility traces × reward prediction error
     * - Reward signal GATES plasticity
     * - Suitable for: action selection, policy learning, control
     * - Biological analog: basal ganglia, prefrontal cortex, motor areas
     */
    REWARD_MODULATED_STDP,
    
    /**
     * Mixed Mode (Hybrid)
     * - Combines both unsupervised and supervised learning
     * - Reward acts as a MODULATOR (not a gate)
     * - Suitable for: executive function, working memory
     * - Biological analog: prefrontal cortex, anterior cingulate
     */
    MIXED_STDP
};

/**
 * @brief Get human-readable name for learning mode
 */
inline const char* learningModeToString(LearningMode mode) {
    switch (mode) {
        case LearningMode::PURE_STDP: return "Pure STDP";
        case LearningMode::REWARD_MODULATED_STDP: return "Reward-Modulated STDP";
        case LearningMode::MIXED_STDP: return "Mixed STDP";
        default: return "Unknown";
    }
}

/**
 * @brief Learning parameters for each mode
 */
struct LearningModeParams {
    LearningMode mode = LearningMode::PURE_STDP;
    
    // STDP parameters
    float stdp_learning_rate = 0.01f;
    float stdp_tau_positive = 20.0f;  // ms, LTP time constant
    float stdp_tau_negative = 20.0f;  // ms, LTD time constant
    float stdp_a_positive = 0.005f;   // LTP amplitude
    float stdp_a_negative = 0.003f;   // LTD amplitude
    
    // Reward modulation parameters
    float reward_sensitivity = 1.0f;   // How much reward affects learning
    float eligibility_decay = 0.95f;   // Eligibility trace decay per timestep
    float baseline_reward = 0.0f;      // Expected reward (for RPE calculation)
    
    // Mixed mode parameters
    float unsupervised_weight = 0.5f;  // Mixing ratio for MIXED_STDP mode
    float supervised_weight = 0.5f;
};

} // namespace neurogen