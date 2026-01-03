#pragma once
#include <vector>
#include <string>
#include <memory>
#include "engine/CorticalColumnV2.h"
#include "engine/GPUNeuralStructures.h" // Added to define GPUNeuronState and GPUSynapse
#include "modules/LearningMode.h"

// Forward declaration ensuring namespace validity
namespace neurogen {
    class NeuralEngine;
}

class CorticalModule {
public:
    struct ModulationParams {
        float dopamine_sensitivity = 0.0f;
        float serotonin_sensitivity = 0.0f;
        float inhibition_level = 0.0f;
        float attention_threshold = 0.0f;
        float excitability_bias = 1.0f;
    };

    struct Config {
        std::string module_name;
        int num_columns = 1;  // Number of cortical columns in this module
        int num_neurons = 1024; // Total neurons (will be divided among columns)
        bool enable_plasticity = true;
        float learning_rate = 0.01f;
        int fanout_per_neuron = 256;
        int num_inputs = 0;
        int num_outputs = 0;
        float lateral_connectivity_density = 0.1f; // Density of inter-column connections
        ModulationParams modulation;

        neurogen::LearningMode learning_mode = neurogen::LearningMode::PURE_STDP;
        neurogen::LearningModeParams learning_params;
        
        // Override flag: if true, reward_modulation is forced to match learning_mode
        bool enforce_learning_mode = true;
        
        /**
        * @brief Configure for pure unsupervised STDP (sensory/memory regions)
        */
        void configurePureSTDP() {
            learning_mode = neurogen::LearningMode::PURE_STDP;
            learning_params.mode = neurogen::LearningMode::PURE_STDP;
            learning_params.reward_sensitivity = 0.0f;  // Completely ignore reward
            enforce_learning_mode = true;
        }
        
        /**
        * @brief Configure for reward-modulated learning (action/control regions)cd D
        */
        void configureRewardModulated(float sensitivity = 1.0f) {
            learning_mode = neurogen::LearningMode::REWARD_MODULATED_STDP;
            learning_params.mode = neurogen::LearningMode::REWARD_MODULATED_STDP;
            learning_params.reward_sensitivity = sensitivity;
            enforce_learning_mode = true;
        }
        
        /**
        * @brief Configure for mixed learning (executive function regions)
        */
        void configureMixedSTDP(float unsupervised_ratio = 0.3f) {
            learning_mode = neurogen::LearningMode::MIXED_STDP;
            learning_params.mode = neurogen::LearningMode::MIXED_STDP;
            learning_params.unsupervised_weight = unsupervised_ratio;
            learning_params.supervised_weight = 1.0f - unsupervised_ratio;
            learning_params.reward_sensitivity = 0.5f;
            enforce_learning_mode = true;
        }
    };

    CorticalModule(const Config& config, int gpu_device_id);
    ~CorticalModule();

    // Optimized: Uses internal buffer
    void receiveInput(const std::vector<float>& input_vector);

    void update(float dt_ms, float reward_signal);

    // Optimized: Returns const reference
    const std::vector<float>& getOutputState() const;

    void setPlasticity(bool enabled);
    void modulate(float dopamine, float serotonin);

    // Signal Processing
    float calculateSignalToNoise(const std::vector<float>& signal) const;
    
    // Novelty Gating (Predictive Coding)
    std::vector<float> gateSignal(const std::vector<float>& signal, float threshold);
    
    // Predictive Coding Support
    std::vector<float> calculatePredictionError(const std::vector<float>& prediction);
    
    // Error-driven Modulation
    void modulateWithError(const std::vector<float>& error_signal);
    
    // Gating Signal
    float getGatingSignal() const;

    void setWorkingMemory(const std::vector<float>& memory_state);
    std::vector<float> getWorkingMemory() const;
    void applyTopDownBias(float bias_strength);
    
    // Reset module state (clear working memory, reset statistics)
    void reset();

    std::string getName() const { return config_.module_name; }
    const Config& getConfig() const { return config_; }

    // Updated to aggregate state from columns
    std::vector<GPUNeuronState> getNeuronStates() const;
    std::vector<GPUSynapse> getSynapseStates() const;
    bool setNeuronStates(const std::vector<GPUNeuronState>& neurons);
    bool setSynapseStates(const std::vector<GPUSynapse>& synapses);

    float getDopamineLevel() const { return current_dopamine_; }
    float getSerotoninLevel() const { return current_serotonin_; }
    void restoreNeuromodulatorLevels(float dopamine, float serotonin);

private:
    Config config_;
    
    // Replaced NeuralEngine with multiple CorticalColumnV2 instances
    std::vector<std::unique_ptr<neurogen::cortical::CorticalColumnV2>> columns_;
    
    // CUDA buffers for module-level I/O
    float* d_module_input_ = nullptr;
    float* d_module_output_ = nullptr;
    size_t input_buffer_size_ = 0;
    size_t output_buffer_size_ = 0;

    // Neuromodulation state
    float current_dopamine_;
    float current_serotonin_;
    float current_inhibition_;
    
    // Signal statistics
    float signal_mean_;
    float signal_variance_;
    
    // Buffers
    std::vector<float> working_memory_;
    std::vector<float> input_buffer_;
    std::vector<float> modulation_buffer_;
    mutable std::vector<float> cached_output_state_;
    std::vector<float> previous_input_;
    
    void updateSignalStatistics(const std::vector<float>& signal);
};