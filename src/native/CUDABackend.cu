#include "PackedBackend.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace neurogen::cortical {
namespace {
void check(cudaError_t r) {
  if (r != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(r));
}
__global__ void compute(const float *x, float *y, Dispatch p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x, n = p.n;
  if (i >= n)
    return;
  if (p.mode == 0) {
    float value = x[n * p.width + p.width + i];
    for (unsigned j = 0; j < p.width; ++j)
      value += x[i * p.width + j] * x[n * p.width + j];
    y[i] = value;
    return;
  }
  unsigned col = i / 64, id = i % 64;
  bool exc = x[6 * n + n * 64 + id] > 0;
  float recurrent = 0;
  for (unsigned j = 0; j < 64; ++j)
    recurrent += x[6 * n + col * 4096 + id * 64 + j] * x[n + col * 64 + j];
  float neighbor = 0;
  if (exc) {
    if (col > 0)
      neighbor += x[n + i - 64];
    if (i + 64 < n)
      neighbor += x[n + i + 64];
  }
  float decay = exc ? .6f : .4f;
  float voltage =
      decay * x[i] + (1 - decay) * (x[5 * n + i] + recurrent + .08f * neighbor);
  float spike = (voltage - (exc ? 1.f : .6f) - x[2 * n + i] >= 0 ? 1.f : 0.f) *
                (1 - x[3 * n + i]);
  y[i] = voltage * (1 - spike);
  y[n + i] = .8f * x[n + i] + .2f * spike;
  y[2 * n + i] = .98f * x[2 * n + i] + (exc ? .03f : 0.f) * spike;
  y[3 * n + i] = spike;
  y[4 * n + i] = x[4 * n + i] + spike;
}
struct Buffer {
  float *p = nullptr;
  explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(float))); }
  ~Buffer() { cudaFree(p); }
};
} // namespace
class CUDABackend final : public PackedBackend {
  std::string label_;
  std::unique_ptr<Buffer> input_, output_;
  size_t input_capacity_ = 0, output_capacity_ = 0;

public:
  CUDABackend() {
    int device;
    check(cudaGetDevice(&device));
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, device));
    label_ = std::string("cuda: ") + p.name;
  }
  std::string name() const override { return label_; }

private:
  Vector dispatch(const Vector &x, size_t n, Dispatch p) override {
    if (x.size() > UINT32_MAX || n > UINT32_MAX)
      throw std::runtime_error(
          "Model operation exceeds CUDA kernel indexing limits");
    if (x.size() > input_capacity_) {
      input_ = std::make_unique<Buffer>(x.size());
      input_capacity_ = x.size();
    }
    if (n > output_capacity_) {
      output_ = std::make_unique<Buffer>(n);
      output_capacity_ = n;
    }
    auto &input = *input_;
    auto &output = *output_;
    check(cudaMemcpy(input.p, x.data(), x.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    compute<<<(p.n + 63) / 64, 64>>>(input.p, output.p, p);
    check(cudaGetLastError());
    Vector y(n);
    check(cudaMemcpy(y.data(), output.p, n * sizeof(float),
                     cudaMemcpyDeviceToHost));
    return y;
  }
};
std::unique_ptr<ComputeBackend> make_cuda() {
  return std::make_unique<CUDABackend>();
}
} // namespace neurogen::cortical
