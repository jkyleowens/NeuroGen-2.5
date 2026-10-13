#pragma once
#include <cstddef>
#include <vector>

namespace neurogen::cortical {
// Restores the old structure-of-arrays organization, with the six state fields
// of the simplified normalized LIF language model (not the retired HH model).
struct ALIFNeuronArrays {
  std::vector<float> voltage, trace, adaptation, last_spikes, counts;
  explicit ALIFNeuronArrays(size_t n = 0)
      : voltage(n), trace(n), adaptation(n), last_spikes(n), counts(n) {}
};
struct LayerParams {
  int layer, excitatory, inhibitory;
};
inline constexpr LayerParams layers[] = {{1, 0, 2},  {2, 8, 2},  {3, 8, 2},
                                         {4, 12, 3}, {5, 12, 3}, {6, 10, 2}};
} // namespace neurogen::cortical
