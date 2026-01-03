#include "modules/InterModuleConnection.h"
#include "modules/CorticalModule.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

InterModuleConnection::InterModuleConnection(const Config& config)
    : config_(config),
      current_strength_(config.initial_strength),
      fast_weight_(0.0f),
      attention_modulation_(1.0f),
      pre_synaptic_trace_(0.0f),
      post_synaptic_trace_(0.0f),
      average_activity_(0.0f),
      total_transmitted_(0.0f),
      activation_count_(0) {
    
    if (!config_.source_module || !config_.target_module) {
        std::cerr << "⚠️  Warning: InterModuleConnection created with null module pointers" << std::endl;
    }
}

float InterModuleConnection::transmit(float dt) {
    (void)dt; // Unused parameter
    
    if (!config_.source_module || !config_.target_module) {
        return 0.0f;
    }

    // 1. Get Input (Reference to avoid copy)
    const std::vector<float>& source_output = config_.source_module->getOutputState();
    
    if (source_output.empty()) {
        return 0.0f;
    }

    // 2. Calculate Source Activity (Mean field)
    float source_activity = std::accumulate(source_output.begin(), source_output.end(), 0.0f) 
                           / source_output.size();

    // 3. === FAST WEIGHT ATTENTION MECHANISM ===
    if (config_.fast_weights.enable) {
        // Decay the previous context (Fading memory)
        fast_weight_ *= config_.fast_weights.decay;
        
        // Hebbian-like update for Fast Weights (Pre-synaptic driven)
        // If the source is active, we temporarily strengthen this bond
        // This binds the current input to the immediate future context
        if (std::abs(source_activity) > 0.001f) {
            float update = std::abs(source_activity) * config_.fast_weights.learning_rate;
            fast_weight_ += update;
            
            // Cap the fast weight attention
            fast_weight_ = std::min(config_.fast_weights.max_strength, fast_weight_);
        }
    }

    // 4. Calculate Effective Strength (Slow + Fast)
    // This is the core of "Linear Attention" in RNNs/SNNs
    float effective_strength = current_strength_ + fast_weight_;
    
    // Apply attention modulation (Top-down control)
    effective_strength *= attention_modulation_;

    // 5. Calculate Signal
    float signal_strength = source_activity * effective_strength;
    
    // Polarity (Excitatory/Inhibitory)
    if (!config_.is_excitatory) {
        effective_strength = -std::abs(effective_strength);
        signal_strength = -std::abs(signal_strength);
    }

    // 6. Gating
    if (std::abs(signal_strength) < config_.gating_threshold) {
        // Soft gating scaling
        float scaling = std::abs(signal_strength) / std::max(1e-6f, config_.gating_threshold);
        effective_strength *= scaling;
        signal_strength *= scaling;
    }

    // 7. Transmit Vector
    if (transmitted_signal_buffer_.size() != source_output.size()) {
        transmitted_signal_buffer_.resize(source_output.size());
    }

    // Vectorized apply
    const float* src = source_output.data();
    float* dst = transmitted_signal_buffer_.data();
    size_t size = source_output.size();

    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i] * effective_strength;
    }

    config_.target_module->receiveInput(transmitted_signal_buffer_);

    // 8. Update Stats
    activation_count_++;
    total_transmitted_ += std::abs(signal_strength);
    average_activity_ = 0.99f * average_activity_ + 0.01f * std::abs(source_activity);
    pre_synaptic_trace_ = TRACE_DECAY * pre_synaptic_trace_ + source_activity;

    return signal_strength;
}

void InterModuleConnection::updatePlasticity(float source_activity, float target_activity,
                                            float reward, float dt) {
    if (!config_.enable_plasticity) {
        return;
    }

    // Update traces
    pre_synaptic_trace_ = TRACE_DECAY * pre_synaptic_trace_ + source_activity;
    post_synaptic_trace_ = TRACE_DECAY * post_synaptic_trace_ + target_activity;

    // Hebbian learning: "neurons that fire together, wire together"
    float hebbian_term = pre_synaptic_trace_ * post_synaptic_trace_;

    // Reward modulation (dopaminergic gating)
    float reward_modulation = 1.0f + reward;

    // Weight update
    float delta_strength = config_.plasticity_rate * hebbian_term * reward_modulation * dt;

    // Apply update to SLOW weights only
    current_strength_ += delta_strength;

    // Bound the connection strength
    current_strength_ = std::max(STRENGTH_BOUNDS_MIN, 
                                std::min(STRENGTH_BOUNDS_MAX, current_strength_));
}

void InterModuleConnection::applyAttentionGating(float attention_strength) {
    attention_modulation_ = std::max(0.0f, std::min(1.0f, attention_strength));
}

void InterModuleConnection::setStrength(float strength) {
    current_strength_ = std::max(STRENGTH_BOUNDS_MIN, 
                                std::min(STRENGTH_BOUNDS_MAX, strength));
}

InterModuleConnection::Stats InterModuleConnection::getStats() const {
    return Stats{
        current_strength_,
        fast_weight_,
        average_activity_,
        total_transmitted_,
        activation_count_,
        attention_modulation_,
        pre_synaptic_trace_,
        post_synaptic_trace_
    };
}

void InterModuleConnection::restoreState(float current_strength,
                                         float attention_modulation,
                                         float pre_synaptic_trace,
                                         float post_synaptic_trace,
                                         float average_activity,
                                         float total_transmitted,
                                         int activation_count) {
    setStrength(current_strength);
    attention_modulation_ = std::max(0.0f, std::min(1.0f, attention_modulation));
    pre_synaptic_trace_ = pre_synaptic_trace;
    post_synaptic_trace_ = post_synaptic_trace;
    average_activity_ = average_activity;
    total_transmitted_ = total_transmitted;
    activation_count_ = activation_count;
    // Fast weights are transient, so we usually don't restore them, 
    // or we reset them to 0.
    fast_weight_ = 0.0f;
}