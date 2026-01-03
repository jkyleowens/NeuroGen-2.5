#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine/GPUNeuralStructures.h"
#include "persistence/CheckpointFormat.h"

namespace persistence {

struct ModuleConfigSnapshot {
    std::string module_name;
    int num_neurons = 0;
    bool enable_plasticity = true;
    float learning_rate = 0.0f;
    int fanout_per_neuron = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    float dopamine_sensitivity = 0.0f;
    float serotonin_sensitivity = 0.0f;
    float inhibition_level = 0.0f;
    float attention_threshold = 0.0f;
    float excitability_bias = 1.0f;
};

struct ModuleSnapshot {
    ModuleConfigSnapshot config;
    uint32_t module_index = 0;
    std::vector<GPUNeuronState> neurons;
    std::vector<GPUSynapse> synapses;
    std::vector<float> working_memory;
    float dopamine_level = 0.0f;
    float serotonin_level = 0.0f;
};

struct ConnectionSnapshot {
    std::string name;
    std::string source_module;
    std::string target_module;
    bool is_excitatory = true;
    bool plasticity_enabled = true;
    float current_strength = 0.0f;
    float gating_threshold = 0.0f;
    float plasticity_rate = 0.0f;
    float attention_modulation = 1.0f;
    float average_activity = 0.0f;
    float total_transmitted = 0.0f;
    int activation_count = 0;
    float pre_synaptic_trace = 0.0f;
    float post_synaptic_trace = 0.0f;
};

struct OptimizerSnapshot {
    std::vector<uint8_t> learning_state_blob;
};

struct RNGSnapshot {
    std::vector<uint64_t> seeds;
};

struct BrainSnapshot {
    uint32_t format_version = kCheckpointFormatVersion;
    uint64_t training_step = 0;
    uint64_t cognitive_cycles = 0;
    uint64_t tokens_processed = 0;
    float average_reward = 0.0f;
    float time_since_consolidation_ms = 0.0f;
    std::vector<ModuleSnapshot> modules;
    std::vector<ConnectionSnapshot> connections;
    OptimizerSnapshot optimizer_state;
    RNGSnapshot rng_state;
};

} // namespace persistence


