#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace persistence {

inline constexpr uint32_t kCheckpointFormatVersion = 1;
inline constexpr uint32_t kSectionDescriptorVersion = 1;
inline constexpr uint32_t kTensorDescriptorVersion = 1;
inline constexpr size_t kMaxTensorRank = 6;
inline constexpr std::array<char, 8> kCheckpointMagic = {'N', 'G', 'C', 'H', 'K', 'P', 'T', '1'};

enum class Endianness : uint8_t {
    Little = 0,
    Big = 1
};

enum class SectionType : uint32_t {
    Metadata = 0,
    Neurons = 1,
    Synapses = 2,
    Optimizer = 3,
    RandomState = 4,
    TrainingHistory = 5
};

enum class TensorElementType : uint16_t {
    Float32 = 0,
    Float16 = 1,
    Float64 = 2,
    Int32 = 3,
    Int64 = 4,
    UInt32 = 5,
    UInt64 = 6,
    Byte = 7
};

struct TensorDescriptor {
    uint32_t version = kTensorDescriptorVersion;
    TensorElementType element_type = TensorElementType::Float32;
    uint32_t rank = 0;
    uint64_t element_count = 0;
    std::array<uint64_t, kMaxTensorRank> dimensions = {0, 0, 0, 0, 0, 0};
};

struct SectionDescriptor {
    uint32_t version = kSectionDescriptorVersion;
    SectionType type = SectionType::Metadata;
    uint32_t flags = 0;
    uint64_t offset_bytes = 0;
    uint64_t length_bytes = 0;
    uint64_t record_count = 0;
};

struct CheckpointHeader {
    std::array<char, 8> magic = kCheckpointMagic;
    uint32_t version = kCheckpointFormatVersion;
    uint32_t header_size = sizeof(CheckpointHeader);
    Endianness endianness = Endianness::Little;
    uint32_t section_count = 0;
    uint64_t creation_timestamp = 0;
    uint64_t training_step = 0;
    uint64_t samples_seen = 0;
    uint64_t rng_seed = 0;
};

struct SectionLayout {
    SectionDescriptor descriptor;
    TensorDescriptor tensor;
};

struct CheckpointLayout {
    CheckpointHeader header;
    std::vector<SectionLayout> sections;
};

Endianness detectHostEndianness();
bool isMagicValid(const std::array<char, 8>& magic);
bool isHeaderCompatible(const CheckpointHeader& header);
size_t elementTypeSize(TensorElementType type);
const char* sectionTypeToString(SectionType type);
const char* tensorElementTypeToString(TensorElementType type);

CheckpointHeader makeHeader(uint32_t section_count,
                            uint64_t training_step,
                            uint64_t samples_seen,
                            uint64_t rng_seed);

} // namespace persistence


