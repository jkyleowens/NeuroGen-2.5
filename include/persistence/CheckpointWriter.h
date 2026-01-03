#pragma once

#include <fstream>
#include <iosfwd>
#include <string>
#include <vector>

#include "persistence/CheckpointFormat.h"
#include "persistence/NetworkSnapshot.h"

namespace persistence {

class CheckpointWriter {
public:
    explicit CheckpointWriter(std::string output_path);
    ~CheckpointWriter();

    // Initialize file and write header placeholder
    bool initialize(const BrainSnapshot& metadata_snapshot);

    // Write the Metadata section
    bool writeMetadataSection(const BrainSnapshot& snapshot);

    // Streaming API for Neurons
    bool beginNeuronSection(uint32_t module_count);
    template<typename T>
    bool writeModuleNeurons(uint32_t module_index, const std::vector<T>& neurons);
    bool endNeuronSection();

    // Streaming API for Synapses
    bool beginSynapseSection(uint32_t module_count);
    template<typename T>
    bool writeModuleSynapses(uint32_t module_index, const std::vector<T>& synapses);
    bool endSynapseSection();

    // Write small sections
    bool writeOptimizerSection(const BrainSnapshot& snapshot);
    bool writeRandomStateSection(const BrainSnapshot& snapshot);

    // Finalize the file (update headers/offsets)
    bool finalize();

private:
    std::string output_path_;
    std::ofstream stream_;
    std::vector<SectionLayout> sections_;
    
    // Internal helper to write primitive types
    template <typename T>
    void writePrimitive(T value) {
        stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    bool ensureParentDirectory() const;
};

// Template implementations must be in the header
template<typename T>
bool CheckpointWriter::writeModuleNeurons(uint32_t module_index, const std::vector<T>& neurons) {
    if (!stream_.is_open()) return false;
    writePrimitive(module_index);
    uint64_t count = static_cast<uint64_t>(neurons.size());
    writePrimitive(count);
    if (count > 0) {
        stream_.write(reinterpret_cast<const char*>(neurons.data()), 
                     static_cast<std::streamsize>(count * sizeof(T)));
    }
    return true;
}

template<typename T>
bool CheckpointWriter::writeModuleSynapses(uint32_t module_index, const std::vector<T>& synapses) {
    return writeModuleNeurons(module_index, synapses); 
}

} // namespace persistence