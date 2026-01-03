#include "persistence/CheckpointReader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace persistence {

namespace {

template <typename T>
bool readPrimitive(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

bool readBool(std::istream& stream, bool& value) {
    uint8_t byte = 0;
    if (!readPrimitive(stream, byte)) {
        return false;
    }
    value = (byte != 0);
    return true;
}

bool readString(std::istream& stream, std::string& value) {
    uint32_t size = 0;
    if (!readPrimitive(stream, size)) {
        return false;
    }
    value.resize(size);
    if (size > 0) {
        stream.read(value.data(), static_cast<std::streamsize>(size));
    }
    return stream.good();
}

bool readFloatVector(std::istream& stream, std::vector<float>& values) {
    uint64_t count = 0;
    if (!readPrimitive(stream, count)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    if (count > 0) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(count * sizeof(float)));
    }
    return stream.good();
}

template <typename T>
bool readPodVector(std::istream& stream, std::vector<T>& values) {
    uint64_t count = 0;
    if (!readPrimitive(stream, count)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    if (count > 0) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(count * sizeof(T)));
    }
    return stream.good();
}

} // namespace

CheckpointReader::CheckpointReader(std::string input_path)
    : input_path_(std::move(input_path)) {}

bool CheckpointReader::read(BrainSnapshot& snapshot) {
    std::ifstream stream(input_path_, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "❌ Failed to open checkpoint: " << input_path_ << std::endl;
        return false;
    }

    CheckpointHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream.good()) {
        std::cerr << "❌ Failed to read checkpoint header: " << input_path_ << std::endl;
        return false;
    }

    if (!isHeaderCompatible(header)) {
        std::cerr << "❌ Incompatible checkpoint format: " << input_path_ << std::endl;
        return false;
    }

    std::vector<SectionDescriptor> descriptors(header.section_count);
    stream.read(reinterpret_cast<char*>(descriptors.data()),
                static_cast<std::streamsize>(descriptors.size() * sizeof(SectionDescriptor)));
    if (!stream.good()) {
        std::cerr << "❌ Failed to read section descriptors: " << input_path_ << std::endl;
        return false;
    }

    snapshot = BrainSnapshot{};
    snapshot.format_version = header.version;
    snapshot.training_step = header.training_step;
    snapshot.tokens_processed = header.samples_seen;

    for (const auto& descriptor : descriptors) {
        stream.seekg(static_cast<std::streamoff>(descriptor.offset_bytes), std::ios::beg);
        switch (descriptor.type) {
            case SectionType::Metadata:
                if (!parseMetadataSection(stream, descriptor, snapshot)) return false;
                break;
            case SectionType::Neurons:
                if (!parseNeuronSection(stream, descriptor, snapshot)) return false;
                break;
            case SectionType::Synapses:
                if (!parseSynapseSection(stream, descriptor, snapshot)) return false;
                break;
            case SectionType::Optimizer:
                if (!parseOptimizerSection(stream, descriptor, snapshot)) return false;
                break;
            case SectionType::RandomState:
                if (!parseRandomStateSection(stream, descriptor, snapshot)) return false;
                break;
            default:
                std::cerr << "⚠️  Unknown checkpoint section ignored." << std::endl;
                break;
        }
    }

    return true;
}

std::optional<BrainSnapshot> CheckpointReader::read() {
    BrainSnapshot snapshot;
    if (read(snapshot)) {
        return snapshot;
    }
    return std::nullopt;
}

bool CheckpointReader::parseMetadataSection(std::istream& stream,
                                            const SectionDescriptor& descriptor,
                                            BrainSnapshot& snapshot) {
    (void)descriptor;
    if (!readPrimitive(stream, snapshot.training_step)) return false;
    if (!readPrimitive(stream, snapshot.cognitive_cycles)) return false;
    if (!readPrimitive(stream, snapshot.tokens_processed)) return false;
    if (!readPrimitive(stream, snapshot.average_reward)) return false;
    if (!readPrimitive(stream, snapshot.time_since_consolidation_ms)) return false;

    uint32_t module_count = 0;
    if (!readPrimitive(stream, module_count)) return false;

    snapshot.modules.clear();
    snapshot.modules.reserve(module_count);

    for (uint32_t i = 0; i < module_count; ++i) {
        ModuleSnapshot module;
        if (!readPrimitive(stream, module.module_index)) return false;
        if (!readString(stream, module.config.module_name)) return false;
        if (!readPrimitive(stream, module.config.num_neurons)) return false;
        if (!readBool(stream, module.config.enable_plasticity)) return false;
        if (!readPrimitive(stream, module.config.learning_rate)) return false;
        if (!readPrimitive(stream, module.config.fanout_per_neuron)) return false;
        if (!readPrimitive(stream, module.config.num_inputs)) return false;
        if (!readPrimitive(stream, module.config.num_outputs)) return false;
        if (!readPrimitive(stream, module.config.dopamine_sensitivity)) return false;
        if (!readPrimitive(stream, module.config.serotonin_sensitivity)) return false;
        if (!readPrimitive(stream, module.config.inhibition_level)) return false;
        if (!readPrimitive(stream, module.config.attention_threshold)) return false;
        if (!readPrimitive(stream, module.config.excitability_bias)) return false;
        if (!readPrimitive(stream, module.dopamine_level)) return false;
        if (!readPrimitive(stream, module.serotonin_level)) return false;
        if (!readFloatVector(stream, module.working_memory)) return false;
        snapshot.modules.push_back(std::move(module));
    }

    uint32_t connection_count = 0;
    if (!readPrimitive(stream, connection_count)) return false;
    snapshot.connections.clear();
    snapshot.connections.reserve(connection_count);

    for (uint32_t i = 0; i < connection_count; ++i) {
        ConnectionSnapshot connection;
        if (!readString(stream, connection.name)) return false;
        if (!readString(stream, connection.source_module)) return false;
        if (!readString(stream, connection.target_module)) return false;
        if (!readBool(stream, connection.is_excitatory)) return false;
        if (!readBool(stream, connection.plasticity_enabled)) return false;
        if (!readPrimitive(stream, connection.current_strength)) return false;
        if (!readPrimitive(stream, connection.gating_threshold)) return false;
        if (!readPrimitive(stream, connection.plasticity_rate)) return false;
        if (!readPrimitive(stream, connection.attention_modulation)) return false;
        if (!readPrimitive(stream, connection.average_activity)) return false;
        if (!readPrimitive(stream, connection.total_transmitted)) return false;
        if (!readPrimitive(stream, connection.activation_count)) return false;
        if (!readPrimitive(stream, connection.pre_synaptic_trace)) return false;
        if (!readPrimitive(stream, connection.post_synaptic_trace)) return false;
        snapshot.connections.push_back(std::move(connection));
    }

    return true;
}

bool CheckpointReader::parseNeuronSection(std::istream& stream,
                                          const SectionDescriptor& descriptor,
                                          BrainSnapshot& snapshot) {
    (void)descriptor;
    uint32_t module_count = 0;
    if (!readPrimitive(stream, module_count)) return false;

    for (uint32_t i = 0; i < module_count; ++i) {
        uint32_t module_index = 0;
        if (!readPrimitive(stream, module_index)) return false;

        ModuleSnapshot* module = findModuleSnapshot(snapshot, module_index);
        if (!module) {
            std::cerr << "⚠️  Unknown module index in neuron section: " << module_index << std::endl;
            return false;
        }

        uint64_t neuron_count = 0;
        if (!readPrimitive(stream, neuron_count)) return false;

        module->neurons.resize(static_cast<size_t>(neuron_count));
        if (neuron_count > 0) {
            stream.read(reinterpret_cast<char*>(module->neurons.data()),
                        static_cast<std::streamsize>(neuron_count * sizeof(GPUNeuronState)));
            if (!stream.good()) return false;
        }
    }

    return true;
}

bool CheckpointReader::parseSynapseSection(std::istream& stream,
                                           const SectionDescriptor& descriptor,
                                           BrainSnapshot& snapshot) {
    (void)descriptor;
    uint32_t module_count = 0;
    if (!readPrimitive(stream, module_count)) return false;

    for (uint32_t i = 0; i < module_count; ++i) {
        uint32_t module_index = 0;
        if (!readPrimitive(stream, module_index)) return false;

        ModuleSnapshot* module = findModuleSnapshot(snapshot, module_index);
        if (!module) {
            std::cerr << "⚠️  Unknown module index in synapse section: " << module_index << std::endl;
            return false;
        }

        uint64_t synapse_count = 0;
        if (!readPrimitive(stream, synapse_count)) return false;

        module->synapses.resize(static_cast<size_t>(synapse_count));
        if (synapse_count > 0) {
            stream.read(reinterpret_cast<char*>(module->synapses.data()),
                        static_cast<std::streamsize>(synapse_count * sizeof(GPUSynapse)));
            if (!stream.good()) return false;
        }
    }

    return true;
}

bool CheckpointReader::parseOptimizerSection(std::istream& stream,
                                             const SectionDescriptor& descriptor,
                                             BrainSnapshot& snapshot) {
    (void)descriptor;
    uint64_t blob_size = 0;
    if (!readPrimitive(stream, blob_size)) return false;
    snapshot.optimizer_state.learning_state_blob.resize(static_cast<size_t>(blob_size));
    if (blob_size > 0) {
        stream.read(reinterpret_cast<char*>(snapshot.optimizer_state.learning_state_blob.data()),
                    static_cast<std::streamsize>(blob_size));
        if (!stream.good()) return false;
    }
    return true;
}

bool CheckpointReader::parseRandomStateSection(std::istream& stream,
                                               const SectionDescriptor& descriptor,
                                               BrainSnapshot& snapshot) {
    (void)descriptor;
    return readPodVector(stream, snapshot.rng_state.seeds);
}

ModuleSnapshot* CheckpointReader::findModuleSnapshot(BrainSnapshot& snapshot, uint32_t module_index) {
    auto it = std::find_if(snapshot.modules.begin(), snapshot.modules.end(),
                           [module_index](const ModuleSnapshot& module) {
                               return module.module_index == module_index;
                           });
    if (it == snapshot.modules.end()) {
        return nullptr;
    }
    return &(*it);
}

} // namespace persistence


