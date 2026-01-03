#pragma once

#include <optional>
#include <string>

#include "persistence/NetworkSnapshot.h"

namespace persistence {

class CheckpointReader {
public:
    explicit CheckpointReader(std::string input_path);

    bool read(BrainSnapshot& snapshot);
    std::optional<BrainSnapshot> read();

private:
    std::string input_path_;

    bool parseMetadataSection(std::istream& stream,
                              const SectionDescriptor& descriptor,
                              BrainSnapshot& snapshot);
    bool parseNeuronSection(std::istream& stream,
                            const SectionDescriptor& descriptor,
                            BrainSnapshot& snapshot);
    bool parseSynapseSection(std::istream& stream,
                             const SectionDescriptor& descriptor,
                             BrainSnapshot& snapshot);
    bool parseOptimizerSection(std::istream& stream,
                               const SectionDescriptor& descriptor,
                               BrainSnapshot& snapshot);
    bool parseRandomStateSection(std::istream& stream,
                                 const SectionDescriptor& descriptor,
                                 BrainSnapshot& snapshot);

    ModuleSnapshot* findModuleSnapshot(BrainSnapshot& snapshot, uint32_t module_index);
};

} // namespace persistence


