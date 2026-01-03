#include "persistence/CheckpointFormat.h"

#include <chrono>
#include <stdexcept>

namespace persistence {

namespace {
Endianness detectEndianness() {
    uint16_t value = 0x1;
    uint8_t* first_byte = reinterpret_cast<uint8_t*>(&value);
    return (*first_byte == 0x1) ? Endianness::Little : Endianness::Big;
}
} // namespace

Endianness detectHostEndianness() {
    static const Endianness cached = detectEndianness();
    return cached;
}

bool isMagicValid(const std::array<char, 8>& magic) {
    return magic == kCheckpointMagic;
}

bool isHeaderCompatible(const CheckpointHeader& header) {
    if (!isMagicValid(header.magic)) {
        return false;
    }

    if (header.version > kCheckpointFormatVersion) {
        return false;
    }

    return true;
}

size_t elementTypeSize(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float16: return 2;
        case TensorElementType::Float32: return 4;
        case TensorElementType::Float64: return 8;
        case TensorElementType::Int32: return 4;
        case TensorElementType::Int64: return 8;
        case TensorElementType::UInt32: return 4;
        case TensorElementType::UInt64: return 8;
        case TensorElementType::Byte: return 1;
    }
    throw std::runtime_error("Unknown TensorElementType");
}

const char* sectionTypeToString(SectionType type) {
    switch (type) {
        case SectionType::Metadata: return "Metadata";
        case SectionType::Neurons: return "Neurons";
        case SectionType::Synapses: return "Synapses";
        case SectionType::Optimizer: return "Optimizer";
        case SectionType::RandomState: return "RandomState";
        case SectionType::TrainingHistory: return "TrainingHistory";
    }
    return "Unknown";
}

const char* tensorElementTypeToString(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float16: return "float16";
        case TensorElementType::Float32: return "float32";
        case TensorElementType::Float64: return "float64";
        case TensorElementType::Int32: return "int32";
        case TensorElementType::Int64: return "int64";
        case TensorElementType::UInt32: return "uint32";
        case TensorElementType::UInt64: return "uint64";
        case TensorElementType::Byte: return "byte";
    }
    return "unknown";
}

CheckpointHeader makeHeader(uint32_t section_count,
                            uint64_t training_step,
                            uint64_t samples_seen,
                            uint64_t rng_seed) {
    CheckpointHeader header;
    header.magic = kCheckpointMagic;
    header.version = kCheckpointFormatVersion;
    header.header_size = sizeof(CheckpointHeader);
    header.section_count = section_count;
    header.endianness = detectHostEndianness();
    header.creation_timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
    header.training_step = training_step;
    header.samples_seen = samples_seen;
    header.rng_seed = rng_seed;
    return header;
}

} // namespace persistence


