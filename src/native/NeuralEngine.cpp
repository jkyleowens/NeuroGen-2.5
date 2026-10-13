#include "engine/NeuralEngine.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace neurogen::cortical {
namespace {
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
Vector take(const float *&data, size_t n) {
  Vector v(data, data + n);
  data += n;
  return v;
}
} // namespace
CorticalColumnV2::CorticalColumnV2(CorticalColumnConfig config)
    : config_(config) {
  for (auto l : layers)
    for (int i = 0; i < l.excitatory + l.inhibitory; ++i) {
      int id = static_cast<int>(sign_.size());
      bool exc = i < l.excitatory;
      sign_.push_back(exc ? 1.f : -1.f);
      layer_.push_back(l.layer);
      if (l.layer == 4 && exc)
        input_ids_.push_back(id);
      if (l.layer >= 5 || l.layer == 1)
        feedback_ids_.push_back(id);
      if (exc)
        output_ids_.push_back(id);
    }
}
size_t CorticalColumnV2::parameter_count() const {
  size_t c = config_.columns, w = config_.width;
  return c * 4096 + c * input_ids_.size() * (w + 1) +
         c * feedback_ids_.size() * w + w * (c * output_ids_.size() * 2 + 1) +
         3 * w;
}
void CorticalColumnV2::load(const float *&data) {
  size_t c = config_.columns, w = config_.width;
  weights_ = take(data, c * 4096);
  const std::pair<int, int> preferred[] = {{4, 2}, {4, 3}, {2, 3},
                                           {3, 5}, {5, 6}, {6, 4}};
  for (size_t col = 0; col < c; ++col)
    for (int post = 0; post < 64; ++post)
      for (int pre = 0; pre < 64; ++pre) {
        bool allowed = layer_[post] == layer_[pre];
        for (auto p : preferred)
          allowed |= layer_[pre] == p.first && layer_[post] == p.second;
        allowed |= sign_[pre] < 0 && std::abs(layer_[post] - layer_[pre]) <= 1;
        float &raw = weights_[col * 4096 + post * 64 + pre];
        raw =
            allowed && post != pre
                ? (std::max(raw, 0.f) + std::log1p(std::exp(-std::abs(raw)))) *
                      sign_[pre]
                : 0.f;
      }
  input_ = take(data, c * input_ids_.size() * w);
  input_bias_ = take(data, c * input_ids_.size());
  feedback_ = take(data, c * feedback_ids_.size() * w);
  output_ = take(data, w * c * output_ids_.size() * 2);
  output_bias_ = take(data, w);
  norm_ = take(data, w);
  norm_bias_ = take(data, w);
  decay_ = take(data, w);
  for (float &d : decay_)
    d = .5f + .499f * sigmoid(d);
}
Vector CorticalColumnV2::step(const Vector &input, ColumnState &state,
                              bool sensory, ComputeBackend &backend,
                              float &activity) const {
  size_t c = config_.columns, w = config_.width;
  Vector current(c * 64);
  for (size_t i = 0; i < current.size(); ++i)
    current[i] = sign_[i % 64] > 0 ? .15f : .35f;
  if (sensory) {
    auto v = backend.matvec(input_, input, input_bias_);
    for (size_t col = 0; col < c; ++col)
      for (size_t j = 0; j < input_ids_.size(); ++j)
        current[col * 64 + input_ids_[j]] =
            3 * sigmoid(v[col * input_ids_.size() + j]);
  }
  auto fb = backend.matvec(feedback_, state.memory, {});
  for (size_t col = 0; col < c; ++col)
    for (size_t j = 0; j < feedback_ids_.size(); ++j)
      current[col * 64 + feedback_ids_[j]] +=
          .5f * std::tanh(fb[col * feedback_ids_.size() + j]);
  activity =
      backend.ticks(state.neurons, current, weights_, sign_, config_.ticks);
  Vector features;
  features.reserve(c * output_ids_.size() * 2);
  for (const auto *field : {&state.neurons.voltage, &state.neurons.trace})
    for (size_t col = 0; col < c; ++col)
      for (int id : output_ids_)
        features.push_back((*field)[col * 64 + id]);
  auto h = backend.matvec(output_, features, output_bias_);
  double mean = 0, var = 0;
  for (size_t i = 0; i < w; ++i) {
    h[i] += .25f * state.memory[i];
    mean += h[i];
  }
  mean /= w;
  for (float value : h)
    var += (value - mean) * (value - mean);
  float inv = 1.f / std::sqrt(var / w + 1e-5);
  for (size_t i = 0; i < w; ++i) {
    h[i] = (h[i] - static_cast<float>(mean)) * inv * norm_[i] + norm_bias_[i];
    state.memory[i] = decay_[i] * state.memory[i] + (1 - decay_[i]) * h[i];
  }
  return h;
}
NeuralEngine::NeuralEngine(EngineConfig c, const float *data, size_t count,
                           const std::string &backend)
    : config(c) {
  if (c.columns < 1 || c.columns > 4096 || c.width < 1 || c.width > 16384 ||
      c.regions < 1 || c.regions > 128 || c.ticks < 1 || c.ticks > 10000 ||
      c.idle_steps < 0 || c.idle_steps > 16)
    throw std::invalid_argument("Invalid native model dimensions");
  for (int r = 0; r < c.regions; ++r)
    regions_.emplace_back(CorticalColumnConfig{c.columns, c.width, c.ticks});
  if (!data || count != parameter_count())
    throw std::invalid_argument(
        "Native weight count does not match architecture");
  if (!std::all_of(data, data + count,
                   [](float x) { return std::isfinite(x); }))
    throw std::invalid_argument("Non-finite weights");
  embedding_ = take(data, 264ull * c.width);
  for (auto &r : regions_)
    r.load(data);
  head_ = take(data, 264ull * c.width);
  head_bias_ = take(data, 264);
  backend_ = make_backend(backend);
}
size_t NeuralEngine::parameter_count() const {
  return 528ull * config.width + 264 +
         config.regions * regions_.front().parameter_count();
}
size_t NeuralEngine::state_size() const {
  return config.regions * (5ull * config.columns * 64 + config.width);
}
std::vector<ColumnState> NeuralEngine::unpack(const float *source) const {
  std::vector<ColumnState> result;
  for (int r = 0; r < config.regions; ++r) {
    result.emplace_back(config.columns * 64, config.width);
    auto &s = result.back();
    // Checkpoint order: voltage, trace, adaptation, last_spikes, memory,
    // counts.
    for (auto *v : {&s.neurons.voltage, &s.neurons.trace, &s.neurons.adaptation,
                    &s.neurons.last_spikes, &s.memory, &s.neurons.counts}) {
      std::copy(source, source + v->size(), v->begin());
      source += v->size();
    }
  }
  return result;
}
void NeuralEngine::pack(const std::vector<ColumnState> &states,
                        float *dest) const {
  for (auto &s : states)
    for (auto *v : {&s.neurons.voltage, &s.neurons.trace, &s.neurons.adaptation,
                    &s.neurons.last_spikes, &s.memory, &s.neurons.counts}) {
      std::copy(v->begin(), v->end(), dest);
      dest += v->size();
    }
}
Vector NeuralEngine::advance(const Vector &input,
                             std::vector<ColumnState> &states, bool idle,
                             float &activity) {
  Vector h = input;
  activity = 0;
  for (size_t r = 0; r < regions_.size(); ++r) {
    float a = 0;
    h = regions_[r].step(h, states[r], !idle || r > 0, *backend_, a);
    activity += a;
  }
  activity /= config.regions;
  return h;
}
Vector NeuralEngine::step(int token, std::vector<ColumnState> &state,
                          Vector &hidden, float &activity) {
  if (token < 0 || token >= 264)
    throw std::invalid_argument("Token must be in 0..263");
  Vector input(embedding_.begin() + token * config.width,
               embedding_.begin() + (token + 1) * config.width);
  hidden = advance(input, state, false, activity);
  for (int i = 0; i < config.idle_steps; ++i) {
    float a = 0;
    hidden = advance(Vector(config.width), state, true, a);
    activity += a;
  }
  activity /= 1 + config.idle_steps;
  return backend_->matvec(head_, hidden, head_bias_);
}
} // namespace neurogen::cortical
namespace {
thread_local std::string error, backend_label;
using neurogen::cortical::NeuralEngine;
NeuralEngine &checked(void *p, const float *state, size_t n) {
  if (!p)
    throw std::invalid_argument("Null native engine");
  auto &e = *static_cast<NeuralEngine *>(p);
  if (!state || n != e.state_size())
    throw std::invalid_argument("Invalid native state size");
  if (!std::all_of(state, state + n, [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument("Non-finite native state");
  return e;
}
} // namespace
extern "C" {
void *ng_create(int c, int w, int r, int t, int idle, const float *weights,
                size_t n, const char *backend) {
  try {
    error.clear();
    return new NeuralEngine({c, w, r, t, idle}, weights, n,
                            backend ? backend : "cpu");
  } catch (const std::exception &ex) {
    error = ex.what();
    return nullptr;
  }
}
void ng_destroy(void *e) { delete static_cast<NeuralEngine *>(e); }
const char *ng_error() { return error.c_str(); }
const char *ng_backend(void *e) {
  if (!e)
    return "invalid";
  backend_label = static_cast<NeuralEngine *>(e)->backend_name();
  return backend_label.c_str();
}
int ng_step(void *p, int token, const float *s, size_t n, float *next,
            float *logits, float *h, float *a) {
  try {
    error.clear();
    auto &e = checked(p, s, n);
    std::lock_guard<std::mutex> lock(e.mutex);
    if (!next || !logits || !h || !a)
      throw std::invalid_argument("Null output");
    auto state = e.unpack(s);
    std::vector<float> hidden;
    auto out = e.step(token, state, hidden, *a);
    e.pack(state, next);
    std::copy(out.begin(), out.end(), logits);
    std::copy(hidden.begin(), hidden.end(), h);
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
int ng_advance(void *p, const float *input, const float *s, size_t n, int idle,
               float *next, float *h, float *a) {
  try {
    error.clear();
    auto &e = checked(p, s, n);
    std::lock_guard<std::mutex> lock(e.mutex);
    if (!input || !next || !h || !a)
      throw std::invalid_argument("Null input/output");
    if (!std::all_of(input, input + e.config.width,
                     [](float v) { return std::isfinite(v); }))
      throw std::invalid_argument("Non-finite input");
    auto state = e.unpack(s);
    auto out = e.advance({input, input + e.config.width}, state, idle != 0, *a);
    e.pack(state, next);
    std::copy(out.begin(), out.end(), h);
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
}
