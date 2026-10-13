#pragma once
#include "ALIFNeuron.h"
#include <cstddef>
#include <memory>
#include <string>

namespace neurogen::cortical {
using Vector = std::vector<float>;
struct CorticalColumnConfig {
  int columns = 1, width = 32, ticks = 2;
};
struct ColumnState {
  ALIFNeuronArrays neurons;
  Vector memory;
  ColumnState(size_t n, size_t w) : neurons(n), memory(w) {}
};
// Small backend boundary: dense projections and synchronous recurrent LIF
// ticks. Each tick reads only the previous state; no in-place cross-neuron
// races.
class ComputeBackend {
public:
  virtual ~ComputeBackend() = default;
  virtual Vector matvec(const Vector &weights, const Vector &x,
                        const Vector &bias) = 0;
  virtual float ticks(ALIFNeuronArrays &, const Vector &current,
                      const Vector &weights, const Vector &sign, int ticks) = 0;
  virtual std::string name() const = 0;
};
std::unique_ptr<ComputeBackend> make_backend(const std::string &name);
class CorticalColumnV2 {
public:
  explicit CorticalColumnV2(CorticalColumnConfig config);
  void load(const float *&data);
  Vector step(const Vector &input, ColumnState &state, bool sensory,
              ComputeBackend &backend, float &activity) const;
  size_t parameter_count() const;

private:
  CorticalColumnConfig config_;
  Vector weights_, input_, input_bias_, feedback_, output_, output_bias_, norm_,
      norm_bias_, decay_;
  Vector sign_;
  std::vector<int> input_ids_, feedback_ids_, output_ids_, layer_;
};
} // namespace neurogen::cortical
