#pragma once
#include <memory>
#include <vector>
#include <string>

// Forward declaration
class CorticalModule;

/**
 * @brief Represents a connection between two cortical modules (the "Brain Bus")
 * * Now supports "Fast Weights" (Short-Term Plasticity) which acts as a 
 * Linear Attention mechanism for holding sequence context.
 */
class InterModuleConnection {
public:
    struct FastWeightParams {
        bool enable = true;
        float decay = 0.9f;          // How fast context fades (0.0 = instant, 1.0 = forever)
        float learning_rate = 0.2f;  // How quickly we "attend" to new associations
        float max_strength = 1.0f;   // Cap on attentional boost
    };

    struct Config {
        std::string connection_name;
        CorticalModule* source_module;
        CorticalModule* target_module;
        float initial_strength;          
        bool is_excitatory;              
        float gating_threshold;          
        float plasticity_rate;           
        bool enable_plasticity;
        FastWeightParams fast_weights; // NEW: Attention parameters
    };

    InterModuleConnection(const Config& config);
    ~InterModuleConnection() = default;

    /**
     * @brief Transfer signal from source to target module
     * Uses (Slow_Weight + Fast_Weight) to transmit signal.
     */
    float transmit(float dt);

    /**
     * @brief Update connection strength (Long-Term Potentiation)
     */
    void updatePlasticity(float source_activity, float target_activity, 
                         float reward, float dt);

    void applyAttentionGating(float attention_strength);

    float getStrength() const { return current_strength_; }
    float getFastWeight() const { return fast_weight_; } // NEW
    
    void setStrength(float strength);
    bool isExcitatory() const { return config_.is_excitatory; }
    const Config& getConfig() const { return config_; }

    float getAttentionModulation() const { return attention_modulation_; }
    float getPreSynapticTrace() const { return pre_synaptic_trace_; }
    float getPostSynapticTrace() const { return post_synaptic_trace_; }

    struct Stats {
        float current_strength;
        float fast_weight; // NEW
        float average_activity;
        float total_transmitted;
        int activation_count;
        float attention_modulation;
        float pre_synaptic_trace;
        float post_synaptic_trace;
    };
    Stats getStats() const;

    void restoreState(float current_strength,
                      float attention_modulation,
                      float pre_synaptic_trace,
                      float post_synaptic_trace,
                      float average_activity,
                      float total_transmitted,
                      int activation_count);

private:
    Config config_;
    
    // Connection state
    float current_strength_; // The "Slow" Weight (Long-term memory)
    float fast_weight_;      // NEW: The "Fast" Weight (Attention / Context)
    float attention_modulation_;
    
    // Hebbian learning traces
    float pre_synaptic_trace_;
    float post_synaptic_trace_;
    
    // Statistics
    float average_activity_;
    float total_transmitted_;
    int activation_count_;
    
    // Reusable buffer
    std::vector<float> transmitted_signal_buffer_;
    
    // Constants
    static constexpr float TRACE_DECAY = 0.95f;
    static constexpr float STRENGTH_BOUNDS_MIN = 0.01f;
    static constexpr float STRENGTH_BOUNDS_MAX = 2.0f;
};