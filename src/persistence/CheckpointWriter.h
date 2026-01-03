#pragma once

#include <iosfwd>
#include <string>

#include "persistence/NetworkSnapshot.h"

namespace persistence {

class CheckpointWriter {
public:
    explicit CheckpointWriter(std::string output_path);

    bool write(const BrainSnapshot& snapshot);

private:
    std::string output_path_;

    bool ensureParentDirectory() const;
    uint64_t writeMetadataSection(std::ostream& stream,
                                  const BrainSnapshot& snapshot,
                                  SectionLayout& layout);
    uint64_t writeNeuronSection(std::ostream& stream,
                                const BrainSnapshot& snapshot,
                                SectionLayout& layout);
    uint64_t writeSynapseSection(std::ostream& stream,
                                 const BrainSnapshot& snapshot,
                                 SectionLayout& layout);
    uint64_t writeOptimizerSection(std::ostream& stream,
                                   const BrainSnapshot& snapshot,
                                   SectionLayout& layout);
    uint64_t writeRandomStateSection(std::ostream& stream,
                                     const BrainSnapshot& snapshot,
                                     SectionLayout& layout);
};

} // namespace persistence


