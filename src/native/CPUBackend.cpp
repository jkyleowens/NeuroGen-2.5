#include "PackedBackend.h"
#include <stdexcept>
namespace neurogen::cortical {
class CPUBackend final : public PackedBackend {
public:
  std::string name() const override { return "cpu"; }

private:
  Vector dispatch(const Vector &x, size_t size, Dispatch p) override {
    Vector y(size);
    size_t n = p.n;
    for (size_t i = 0; i < n; ++i) {
      if (p.mode == 0) {
        float value = x[n * p.width + p.width + i];
        for (size_t j = 0; j < p.width; ++j)
          value += x[i * p.width + j] * x[n * p.width + j];
        y[i] = value;
        continue;
      }
      size_t col = i / 64, id = i % 64;
      bool exc = x[6 * n + n * 64 + id] > 0;
      float recurrent = 0;
      for (size_t j = 0; j < 64; ++j)
        recurrent += x[6 * n + col * 4096 + id * 64 + j] * x[n + col * 64 + j];
      float neighbor = 0;
      if (exc) {
        if (col > 0)
          neighbor += x[n + i - 64];
        if (i + 64 < n)
          neighbor += x[n + i + 64];
      }
      float decay = exc ? .6f : .4f;
      float voltage = decay * x[i] + (1 - decay) * (x[5 * n + i] + recurrent +
                                                    .08f * neighbor);
      float spike =
          (voltage - (exc ? 1.f : .6f) - x[2 * n + i] >= 0 ? 1.f : 0.f) *
          (1 - x[3 * n + i]);
      y[i] = voltage * (1 - spike);
      y[n + i] = .8f * x[n + i] + .2f * spike;
      y[2 * n + i] = .98f * x[2 * n + i] + (exc ? .03f : 0.f) * spike;
      y[3 * n + i] = spike;
      y[4 * n + i] = x[4 * n + i] + spike;
    }
    return y;
  }
};
#ifdef NEUROGEN_VULKAN
std::unique_ptr<ComputeBackend> make_vulkan();
#endif
#ifdef NEUROGEN_CUDA
std::unique_ptr<ComputeBackend> make_cuda();
#endif
std::unique_ptr<ComputeBackend> make_backend(const std::string &name) {
  if (name == "cpu")
    return std::make_unique<CPUBackend>();
#ifdef NEUROGEN_VULKAN
  if (name == "vulkan")
    return make_vulkan();
#endif
#ifdef NEUROGEN_CUDA
  if (name == "cuda")
    return make_cuda();
#endif
  throw std::invalid_argument("Native backend '" + name +
                              "' unavailable; enable its CMake option and "
                              "rebuild (no silent fallback)");
}
} // namespace neurogen::cortical
