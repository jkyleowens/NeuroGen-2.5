#include <modules/CorticalModule.h>
#include "engine/NetworkConfig.h"
#include "engine/CorticalColumnV2.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

CorticalModule::CorticalModule(const Config& config, int gpu_device_id) 
    : config_(config),
      current_dopamine_(0.0f),
      current_serotonin_(0.0f),
      current_inhibition_(config.modulation.inhibition_level),
      signal_mean_(0.0f),
      signal_variance_(0.0f) {
    
    // Initialize CUDA buffers
    input_buffer_size_ = config.num_inputs * sizeof(float);
    output_buffer_size_ = config.num_outputs * sizeof(float);
    
    if (input_buffer_size_ > 0) {
        cudaMalloc(&d_module_input_, input_buffer_size_);
    }
    if (output_buffer_size_ > 0) {
        cudaMalloc(&d_module_output_, output_buffer_size_);
    }

    // Initialize columns
    int neurons_per_column = config.num_neurons / std::max(1, config.num_columns);
    
    for (int i = 0; i < config.num_columns; ++i) {
        neurogen::cortical::CorticalColumnConfig col_config;
        col_config.name = config.module_name + "_col_" + std::to_string(i);
        col_config.total_neurons = neurons_per_column;
        col_config.column_index = i;
        col_config.gpu_device = gpu_device_id;
        col_config.enable_stdp = config.enable_plasticity;
        col_config.global_learning_rate = config.learning_rate;
        
        auto column = std::make_unique<neurogen::cortical::CorticalColumnV2>(col_config);
        if (column->initialize() != cudaSuccess) {
            std::cerr << "Failed to initialize column " << i << " in module " << config.module_name << std::endl;
        }
        columns_.push_back(std::move(column));
    }
    
    // Pre-allocate buffers to avoid runtime reallocation
    working_memory_.resize(config.num_neurons / 10, 0.0f);
    input_buffer_.reserve(config.num_inputs);
    modulation_buffer_.resize(config.num_inputs);
    cached_output_state_.resize(config.num_outputs);
    previous_input_.resize(config.num_inputs, 0.0f); // Initialize previous input
    
    std::cout << "✓ Initialized module (NeuroGen 2.0): " << config_.module_name 
              << " with " << config.num_columns << " columns (" << neurons_per_column << " neurons/col)" << std::endl;
}

CorticalModule::~CorticalModule() {
    if (d_module_input_) cudaFree(d_module_input_);
    if (d_module_output_) cudaFree(d_module_output_);
}

void CorticalModule::receiveInput(const std::vector<float>& input_vector) {
    // Store input for processing
    input_buffer_ = input_vector;
    
    // Update signal statistics for gating
    updateSignalStatistics(input_vector);
    
    // Apply neuromodulation to inputs
    std::vector<float> modulated_input = input_vector;
    for (size_t i = 0; i < modulated_input.size(); ++i) {
        // Dopamine increases signal strength, serotonin provides stability
        float modulation = 1.0f + (current_dopamine_ * config_.modulation.dopamine_sensitivity);
        modulation *= (1.0f + current_serotonin_ * config_.modulation.serotonin_sensitivity * 0.5f);
        modulation *= (1.0f - current_inhibition_);
        modulated_input[i] *= modulation;
    }
    
    // Copy input to GPU
    if (d_module_input_ && !modulated_input.empty()) {
        cudaMemcpy(d_module_input_, modulated_input.data(), 
                   std::min(input_buffer_size_, modulated_input.size() * sizeof(float)), 
                   cudaMemcpyHostToDevice);
                   
        // Distribute input to columns (L4 feedforward)
        // For now, we broadcast input to all columns or split it. 
        // Assuming split input for now if input size matches total L4 size, otherwise broadcast.
        // Simplified: Broadcast to all columns for robustness
        for (auto& col : columns_) {
            // Note: In a real implementation, we would map specific inputs to specific columns
            // Here we pass the pointer to the module input buffer
            // The column will read what it needs (assumes column knows its input mapping)
            // For this V2 refactor, we'll just pass the whole buffer and let the column handle it
            // But CorticalColumnV2::injectFeedforwardInput expects a buffer of size L4 neurons.
            
            // TODO: Implement proper input mapping. For now, we just inject into L4.
            // We need to know how many L4 neurons are in the column.
            int l4_neurons = col->getNeuronsInLayer(neurogen::cortical::CorticalLayer::L4);
            col->injectFeedforwardInput(d_module_input_, l4_neurons); 
        }
    }
}

void CorticalModule::update(float dt_ms, float reward_signal) {
    // Determine effective reward signal based on learning mode
    float effective_reward = reward_signal;
    
    if (config_.enable_plasticity) {
        switch (config_.learning_mode) {
            case neurogen::LearningMode::PURE_STDP:
                // CRITICAL: Force reward to zero for pure STDP regions
                effective_reward = 0.0f;
                break;
                
            case neurogen::LearningMode::REWARD_MODULATED_STDP:
                // Use reward as-is, scaled by sensitivity
                effective_reward = reward_signal * config_.learning_params.reward_sensitivity;
                break;
                
            case neurogen::LearningMode::MIXED_STDP:
                // Scale reward for mixed mode
                effective_reward = reward_signal * config_.learning_params.reward_sensitivity;
                break;
        }
    } else {
        // Plasticity disabled: no reward signal
        effective_reward = 0.0f;
    }
    
    // Apply neuromodulation effect on effective reward signal
    float modulated_reward = effective_reward * (1.0f + current_dopamine_ * config_.modulation.dopamine_sensitivity);
    
    // Update all columns
    // TODO: Implement lateral connectivity updates here (spike exchange between columns)
    
    for (auto& col : columns_) {
        // Apply STDP if enabled
        if (config_.enable_plasticity) {
            col->applySTDP(modulated_reward);
        }
        
        // Modulate learning
        col->modulateLearning(current_dopamine_, 0.0f, current_serotonin_); // Acetylcholine not yet modeled at module level
        
        // Step the column physics
        col->step(dt_ms, 0.0f); // current_time not tracked here yet
    }
    
    // Decay neuromodulators gradually
    current_dopamine_ *= 0.95f;  // 5% decay per update
    current_serotonin_ *= 0.98f;  // 2% decay per update
}

const std::vector<float>& CorticalModule::getOutputState() const {
    // Aggregate outputs from all columns
    // We collect L5 outputs from each column
    
    size_t total_output_size = 0;
    std::vector<float> temp_output;
    
    for (const auto& col : columns_) {
        int l5_neurons = col->getNeuronsInLayer(neurogen::cortical::CorticalLayer::L5);
        std::vector<float> col_output(l5_neurons);
        
        // Allocate device buffer for this column's output if needed, or reuse a shared one
        // For simplicity, we'll allocate a small temporary buffer on GPU or assume getOutput handles it.
        // CorticalColumnV2::getOutput takes a device pointer.
        
        float* d_col_output;
        cudaMalloc(&d_col_output, l5_neurons * sizeof(float));
        col->getOutput(d_col_output, l5_neurons);
        
        cudaMemcpy(col_output.data(), d_col_output, l5_neurons * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_col_output);
        
        temp_output.insert(temp_output.end(), col_output.begin(), col_output.end());
    }
    
    // Resize cached output if needed
    if (cached_output_state_.size() != temp_output.size()) {
        cached_output_state_.resize(temp_output.size());
    }
    
    cached_output_state_ = temp_output;
    return cached_output_state_;
}

void CorticalModule::modulate(float dopamine, float serotonin) {
    // Apply neuromodulatory signals with sensitivity scaling
    current_dopamine_ += dopamine * config_.modulation.dopamine_sensitivity;
    current_serotonin_ += serotonin * config_.modulation.serotonin_sensitivity;
    
    // Clamp values to reasonable ranges
    current_dopamine_ = std::max(0.0f, std::min(2.0f, current_dopamine_));
    current_serotonin_ = std::max(0.0f, std::min(2.0f, current_serotonin_));
}

float CorticalModule::calculateSignalToNoise(const std::vector<float>& signal) const {
    if (signal.empty()) return 0.0f;
    
    // Calculate mean
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    
    // Calculate variance
    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= signal.size();
    
    // Signal-to-noise ratio: mean / std_dev
    float std_dev = std::sqrt(variance);
    if (std_dev < 1e-6f) return 0.0f;
    
    return std::abs(mean) / std_dev;
}

std::vector<float> CorticalModule::gateSignal(const std::vector<float>& signal, float threshold) {
    std::vector<float> gated_signal(signal.size());
    
    // Ensure previous_input_ is correctly sized (safety check)
    if (previous_input_.size() != signal.size()) {
        previous_input_.resize(signal.size(), 0.0f);
    }

    for (size_t i = 0; i < signal.size(); ++i) {
        float delta = signal[i] - previous_input_[i];
        // If the signal hasn't changed significantly, don't pass it (Novelty Gating)
        if (std::abs(delta) > threshold) {
            gated_signal[i] = delta; 
        } else {
            gated_signal[i] = 0.0f;
        }
        // Update previous input with decay
        previous_input_[i] = signal[i]; 
    }
    
    return gated_signal;
}

void CorticalModule::setWorkingMemory(const std::vector<float>& memory_state) {
    working_memory_ = memory_state;
    if (working_memory_.size() != static_cast<size_t>(config_.num_neurons / 10)) {
        working_memory_.resize(config_.num_neurons / 10, 0.0f);
    }
}

std::vector<float> CorticalModule::getWorkingMemory() const {
    return working_memory_;
}

void CorticalModule::applyTopDownBias(float bias_strength) {
    // Apply a global bias to neuron excitability
    current_inhibition_ = config_.modulation.inhibition_level * (1.0f - bias_strength);
}

void CorticalModule::updateSignalStatistics(const std::vector<float>& signal) {
    if (signal.empty()) return;
    
    // Calculate running average of signal statistics
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    signal_mean_ = 0.9f * signal_mean_ + 0.1f * mean;
    
    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= signal.size();
    signal_variance_ = 0.9f * signal_variance_ + 0.1f * variance;
}

// Removed NeuralEngine accessors as we now use CorticalColumnV2
// neurogen::NeuralEngine* CorticalModule::getNeuralEngine() {
//     return neural_engine_.get();
// }

// const neurogen::NeuralEngine* CorticalModule::getNeuralEngine() const {
//     return neural_engine_.get();
// }

std::vector<GPUNeuronState> CorticalModule::getNeuronStates() const {
    // Aggregate states from all columns
    std::vector<GPUNeuronState> all_states;
    // Note: CorticalColumnV2 doesn't expose getNeuronStates yet.
    // For now, return empty or implement aggregation if CorticalColumnV2 supports it.
    // Assuming we need to implement this later or stub it out.
    return all_states;
}

std::vector<GPUSynapse> CorticalModule::getSynapseStates() const {
    // Aggregate states from all columns
    std::vector<GPUSynapse> all_states;
    // Note: CorticalColumnV2 doesn't expose getSynapseStates yet.
    return all_states;
}

bool CorticalModule::setNeuronStates(const std::vector<GPUNeuronState>& neurons) {
    // Distribute states to columns
    // This is complex because we need to know which neuron belongs to which column.
    // For now, return false as not implemented.
    return false;
}

bool CorticalModule::setSynapseStates(const std::vector<GPUSynapse>& synapses) {
    // Distribute states to columns
    return false;
}

void CorticalModule::restoreNeuromodulatorLevels(float dopamine, float serotonin) {
    current_dopamine_ = std::max(0.0f, std::min(2.0f, dopamine));
    current_serotonin_ = std::max(0.0f, std::min(2.0f, serotonin));
}

std::vector<float> CorticalModule::calculatePredictionError(const std::vector<float>& prediction) {
    // Ideally compare against last received input
    if (input_buffer_.empty() || prediction.empty()) {
        return input_buffer_; // Everything is error/novel if we have no context
    }
    
    size_t size = std::min(input_buffer_.size(), prediction.size());
    std::vector<float> error(size);
    
    for (size_t i = 0; i < size; ++i) {
        error[i] = std::abs(input_buffer_[i] - prediction[i]);
    }
    
    return error;
}

void CorticalModule::modulateWithError(const std::vector<float>& error_signal) {
    if (error_signal.empty()) return;
    
    // Calculate mean absolute error
    float mae = 0.0f;
    for (float e : error_signal) mae += std::abs(e);
    mae /= error_signal.size();
    
    // Boost dopamine locally. High Error = Need to Learn = High Dopamine.
    // Note: In biology, low dopamine often signals error (dip), but for STDP 
    // we often use dopamine to gate the magnitude of change.
    float error_sensitivity = 2.0f;
    current_dopamine_ += mae * error_sensitivity;
    current_dopamine_ = std::min(2.0f, current_dopamine_);
}

float CorticalModule::getGatingSignal() const {
    const auto& output = getOutputState();
    if (output.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (float val : output) sum += val;
    float avg = sum / output.size();
    
    // Sigmoid-like squashing or simple clamp
    return std::min(1.0f, avg * 10.0f); // *10 assuming sparse activity
}

/**
 * Reset module state
 * 
 * Clears:
 * - Working memory buffer
 * - Previous input cache
 * - Signal statistics (mean/variance)
 * - Input/output buffers
 * 
 * Does NOT reset:
 * - Neuron states (handled by NeuralEngine)
 * - Synapse weights (preserved across sequences)
 * - Neuromodulator levels (decay naturally)
 */
void CorticalModule::reset() {
    // Clear buffers
    std::fill(working_memory_.begin(), working_memory_.end(), 0.0f);
    std::fill(previous_input_.begin(), previous_input_.end(), 0.0f);
    input_buffer_.clear();
    
    // Reset output cache
    std::fill(cached_output_state_.begin(), cached_output_state_.end(), 0.0f);
    
    // Reset signal statistics
    signal_mean_ = 0.0f;
    signal_variance_ = 0.0f;
    
    // Note: Neural engine state (neurons, synapses) is NOT reset
    // This preserves learned weights while clearing temporal context
}
