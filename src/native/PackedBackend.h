#pragma once
#include "engine/CorticalColumnV2.h"
#include <algorithm>
#include <cstdint>
namespace neurogen::cortical {
// One backend-neutral wire layout, also used by GLSL/CUDA kernels.
struct Dispatch {
  uint32_t mode, n, width, unused = 0;
};
class PackedBackend : public ComputeBackend {
protected:
  virtual Vector dispatch(const Vector &input, size_t output_size,
                          Dispatch params) = 0;

public:
  Vector matvec(const Vector &weights, const Vector &x,
                const Vector &bias) final {
    size_t rows = weights.size() / x.size();
    Vector input = weights;
    input.insert(input.end(), x.begin(), x.end());
    if (bias.empty())
      input.resize(input.size() + rows, 0);
    else
      input.insert(input.end(), bias.begin(), bias.end());
    return dispatch(
        input, rows,
        {0, static_cast<uint32_t>(rows), static_cast<uint32_t>(x.size())});
  }
  float ticks(ALIFNeuronArrays &s, const Vector &current, const Vector &weights,
              const Vector &sign, int count) final {
    size_t n = s.voltage.size();
    double activity = 0;
    for (int tick = 0; tick < count; ++tick) {
      Vector input;
      input.reserve(6 * n + weights.size() + 64);
      for (auto *v :
           {&s.voltage, &s.trace, &s.adaptation, &s.last_spikes, &s.counts})
        input.insert(input.end(), v->begin(), v->end());
      input.insert(input.end(), current.begin(), current.end());
      input.insert(input.end(), weights.begin(), weights.end());
      input.insert(input.end(), sign.begin(), sign.end());
      auto output = dispatch(input, 5 * n, {1, static_cast<uint32_t>(n), 0});
      size_t offset = 0;
      for (auto *v :
           {&s.voltage, &s.trace, &s.adaptation, &s.last_spikes, &s.counts}) {
        std::copy(output.begin() + offset, output.begin() + offset + n,
                  v->begin());
        offset += n;
      }
      for (float spike : s.last_spikes)
        activity += spike;
    }
    return static_cast<float>(activity / (n * count));
  }
};
} // namespace neurogen::cortical
