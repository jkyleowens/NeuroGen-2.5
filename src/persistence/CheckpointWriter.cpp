#include "persistence/CheckpointWriter.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>
#include <vector>

namespace persistence {

namespace {

// Helper functions for writing to an arbitrary stream
template <typename T>
void writePrimitiveToStream(std::ostream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void writeBool(std::ostream& stream, bool value) {
    uint8_t as_byte = value ? 1u : 0u;
    writePrimitiveToStream(stream, as_byte);
}

void writeString(std::ostream& stream, const std::string& value) {
    uint32_t size = static_cast<uint32_t>(value.size());
    writePrimitiveToStream(stream, size);
    if (size > 0) {
        stream.write(value.data(), static_cast<std::streamsize>(size));
    }
}

void writeFloatVector(std::ostream& stream, const std::vector<float>& values) {
    uint64_t count = static_cast<uint64_t>(values.size());
    writePrimitiveToStream(stream, count);
    if (count > 0) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(count * sizeof(float)));
    }
}

template <typename T>
void writePodVector(std::ostream& stream, const std::vector<T>& values) {
    uint64_t count = static_cast<uint64_t>(values.size());
    writePrimitiveToStream(stream, count);
    if (count > 0) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(count * sizeof(T)));
    }
}

} // namespace

CheckpointWriter::CheckpointWriter(std::string output_path)
    : output_path_(std::move(output_path)) {
    // Initialize section descriptors
    sections_.resize(5);
    sections_[0].descriptor.type = SectionType::Metadata;
    sections_[1].descriptor.type = SectionType::Neurons;
    sections_[2].descriptor.type = SectionType::Synapses;
    sections_[3].descriptor.type = SectionType::Optimizer;
    sections_[4].descriptor.type = SectionType::RandomState;
}

CheckpointWriter::~CheckpointWriter() {
    if (stream_.is_open()) {
        stream_.close();
    }
}

bool CheckpointWriter::ensureParentDirectory() const {
    namespace fs = std::filesystem;
    fs::path path(output_path_);
    fs::path parent = path.parent_path();

    if (parent.empty()) {
        return true;
    }

    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec && !fs::exists(parent)) {
        std::cerr << "❌ Failed to create checkpoint directory \"" << parent
                  << "\": " << ec.message() << std::endl;
        return false;
    }
    return true;
}

bool CheckpointWriter::initialize(const BrainSnapshot& snapshot) {
    if (!ensureParentDirectory()) {
        return false;
    }

    stream_.open(output_path_, std::ios::binary | std::ios::trunc);
    if (!stream_.is_open()) {
        std::cerr << "❌ Failed to open checkpoint file for writing: "
                  << output_path_ << std::endl;
        return false;
    }

    uint64_t rng_seed = snapshot.rng_state.seeds.empty()
        ? snapshot.training_step
        : snapshot.rng_state.seeds.front();

    // Write header with placeholders
    CheckpointHeader header = makeHeader(static_cast<uint32_t>(sections_.size()),
                                         snapshot.training_step,
                                         snapshot.tokens_processed,
                                         rng_seed);

    stream_.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write empty descriptors (to be filled at finalize)
    std::vector<char> descriptor_padding(sections_.size() * sizeof(SectionDescriptor), 0);
    stream_.write(descriptor_padding.data(), static_cast<std::streamsize>(descriptor_padding.size()));

    return stream_.good();
}

bool CheckpointWriter::writeMetadataSection(const BrainSnapshot& snapshot) {
    sections_[0].descriptor.offset_bytes = static_cast<uint64_t>(stream_.tellp());

    writePrimitive(snapshot.training_step);
    writePrimitive(snapshot.cognitive_cycles);
    writePrimitive(snapshot.tokens_processed);
    writePrimitive(snapshot.average_reward);
    writePrimitive(snapshot.time_since_consolidation_ms);

    uint32_t module_count = static_cast<uint32_t>(snapshot.modules.size());
    writePrimitive(module_count);

    for (const auto& module : snapshot.modules) {
        writePrimitive(module.module_index);
        writeString(stream_, module.config.module_name);
        writePrimitive(module.config.num_neurons);
        writeBool(stream_, module.config.enable_plasticity);
        writePrimitive(module.config.learning_rate);
        writePrimitive(module.config.fanout_per_neuron);
        writePrimitive(module.config.num_inputs);
        writePrimitive(module.config.num_outputs);
        writePrimitive(module.config.dopamine_sensitivity);
        writePrimitive(module.config.serotonin_sensitivity);
        writePrimitive(module.config.inhibition_level);
        writePrimitive(module.config.attention_threshold);
        writePrimitive(module.config.excitability_bias);
        writePrimitive(module.dopamine_level);
        writePrimitive(module.serotonin_level);
        writeFloatVector(stream_, module.working_memory);
    }

    uint32_t connection_count = static_cast<uint32_t>(snapshot.connections.size());
    writePrimitive(connection_count);

    for (const auto& connection : snapshot.connections) {
        writeString(stream_, connection.name);
        writeString(stream_, connection.source_module);
        writeString(stream_, connection.target_module);
        writeBool(stream_, connection.is_excitatory);
        writeBool(stream_, connection.plasticity_enabled);
        writePrimitive(connection.current_strength);
        writePrimitive(connection.gating_threshold);
        writePrimitive(connection.plasticity_rate);
        writePrimitive(connection.attention_modulation);
        writePrimitive(connection.average_activity);
        writePrimitive(connection.total_transmitted);
        writePrimitive(connection.activation_count);
        writePrimitive(connection.pre_synaptic_trace);
        writePrimitive(connection.post_synaptic_trace);
    }

    sections_[0].descriptor.record_count = snapshot.modules.size();
    sections_[0].descriptor.length_bytes = static_cast<uint64_t>(stream_.tellp()) - sections_[0].descriptor.offset_bytes;
    
    return stream_.good();
}

bool CheckpointWriter::beginNeuronSection(uint32_t module_count) {
    sections_[1].descriptor.offset_bytes = static_cast<uint64_t>(stream_.tellp());
    writePrimitive(module_count);
    sections_[1].descriptor.record_count = 0; // Will be updated if we tracked total neurons, but distinct from module count
    return stream_.good();
}

bool CheckpointWriter::endNeuronSection() {
    sections_[1].descriptor.length_bytes = static_cast<uint64_t>(stream_.tellp()) - sections_[1].descriptor.offset_bytes;
    return stream_.good();
}

bool CheckpointWriter::beginSynapseSection(uint32_t module_count) {
    sections_[2].descriptor.offset_bytes = static_cast<uint64_t>(stream_.tellp());
    writePrimitive(module_count);
    sections_[2].descriptor.record_count = 0;
    return stream_.good();
}

bool CheckpointWriter::endSynapseSection() {
    sections_[2].descriptor.length_bytes = static_cast<uint64_t>(stream_.tellp()) - sections_[2].descriptor.offset_bytes;
    return stream_.good();
}

bool CheckpointWriter::writeOptimizerSection(const BrainSnapshot& snapshot) {
    sections_[3].descriptor.offset_bytes = static_cast<uint64_t>(stream_.tellp());
    
    const auto& blob = snapshot.optimizer_state.learning_state_blob;
    writePrimitive(static_cast<uint64_t>(blob.size()));
    if (!blob.empty()) {
        stream_.write(reinterpret_cast<const char*>(blob.data()),
                     static_cast<std::streamsize>(blob.size()));
    }
    
    sections_[3].descriptor.record_count = blob.empty() ? 0 : 1;
    sections_[3].descriptor.length_bytes = static_cast<uint64_t>(stream_.tellp()) - sections_[3].descriptor.offset_bytes;
    return stream_.good();
}

bool CheckpointWriter::writeRandomStateSection(const BrainSnapshot& snapshot) {
    sections_[4].descriptor.offset_bytes = static_cast<uint64_t>(stream_.tellp());
    
    writePodVector(stream_, snapshot.rng_state.seeds);
    
    sections_[4].descriptor.record_count = snapshot.rng_state.seeds.size();
    sections_[4].descriptor.length_bytes = static_cast<uint64_t>(stream_.tellp()) - sections_[4].descriptor.offset_bytes;
    return stream_.good();
}

bool CheckpointWriter::finalize() {
    if (!stream_.good()) {
        std::cerr << "❌ Error occurred while writing checkpoint payload: "
                  << output_path_ << std::endl;
        return false;
    }

    // Go back to the beginning to write the real section descriptors
    stream_.seekp(sizeof(CheckpointHeader), std::ios::beg);
    for (const auto& section : sections_) {
        stream_.write(reinterpret_cast<const char*>(&section.descriptor),
                     sizeof(section.descriptor));
    }

    stream_.flush();
    if (!stream_.good()) {
        std::cerr << "❌ Failed to finalize checkpoint file: " << output_path_ << std::endl;
        return false;
    }

    std::cout << "💾 Saved checkpoint to " << output_path_ << std::endl;
    return true;
}

} // namespace persistence