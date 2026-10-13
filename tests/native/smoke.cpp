#include "engine/NeuralEngine.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
  using namespace neurogen::cortical;
  try {
    EngineConfig c{2, 8, 2, 3, 1};
    CorticalColumnV2 column({c.columns, c.width, c.ticks});
    Vector weights(528 * c.width + 264 + c.regions * column.parameter_count(),
                   .1f);
    NeuralEngine cpu(c, weights.data(), weights.size(), "cpu"),
        other(c, weights.data(), weights.size(), argc > 1 ? argv[1] : "cpu");
    Vector zero(cpu.state_size());
    auto a = cpu.unpack(zero.data()), b = other.unpack(zero.data());
    for (int i = 0; i < 12; ++i) {
      Vector ha, hb;
      float aa, ab;
      auto la = cpu.step(i, a, ha, aa), lb = other.step(i, b, hb, ab);
      Vector sa(zero.size()), sb(zero.size());
      cpu.pack(a, sa.data());
      other.pack(b, sb.data());
      for (size_t j = 0; j < sa.size(); ++j)
        if (!std::isfinite(sb[j]) || std::abs(sa[j] - sb[j]) > 1e-4)
          throw std::runtime_error("state mismatch");
      for (size_t j = 0; j < la.size(); ++j)
        if (std::abs(la[j] - lb[j]) > 1e-4)
          throw std::runtime_error("logit mismatch");
    }
    std::cout << "Passed: " << other.backend_name() << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
