#pragma once
#include "CorticalColumnV2.h"
#include <cstdint>
#include <mutex>
namespace neurogen::cortical {
struct EngineConfig {
  int columns, width, regions, ticks, idle_steps;
};
// Regions retain the historical engine -> columns -> neuron arrays structure.
class NeuralEngine {
public:
  NeuralEngine(EngineConfig config, const float *weights, size_t count,
               const std::string &backend);
  size_t state_size() const;
  size_t parameter_count() const;
  Vector advance(const Vector &input, std::vector<ColumnState> &state,
                 bool idle, float &activity);
  Vector step(int token, std::vector<ColumnState> &state, Vector &hidden,
              float &activity);
  std::vector<ColumnState> unpack(const float *state) const;
  void pack(const std::vector<ColumnState> &state, float *output) const;
  EngineConfig config;
  std::string backend_name() const { return backend_->name(); }
  std::mutex mutex;

private:
  std::unique_ptr<ComputeBackend> backend_;
  std::vector<CorticalColumnV2> regions_;
  Vector embedding_, head_, head_bias_;
};
} // namespace neurogen::cortical
// C ABI has no dependency on Python, PyTorch, CUDA headers, or a C++ ABI in
// callers.
extern "C" {
void *ng_create(int columns, int width, int regions, int ticks, int idle_steps,
                const float *weights, size_t count, const char *backend);
void ng_destroy(void *engine);
const char *ng_error();
const char *ng_backend(void *engine);
int ng_step(void *engine, int token, const float *state, size_t state_count,
            float *next_state, float *logits, float *hidden, float *activity);
int ng_advance(void *engine, const float *input, const float *state,
               size_t state_count, int idle, float *next_state, float *hidden,
               float *activity);
}
