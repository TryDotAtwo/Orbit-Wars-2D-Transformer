#include "orbit_wars_cuda.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr size_t kGameCount = 4;
constexpr size_t kRowCount = 64;
constexpr size_t kFeatureCount = 2;
constexpr size_t kLayerCount = 2;
constexpr size_t kDModel = 32;
constexpr size_t kHeadCount = 4;

size_t weight_count() {
  return kFeatureCount * kDModel + kDModel + kLayerCount * kDModel + kDModel * kFeatureCount;
}

int require_ok(OrbitWarsCudaStatus status, const char* operation) {
  if (status.code == 0) {
    return 0;
  }
  std::cerr << "operation=" << operation << "; code=" << status.code
            << "; message=" << status.message << "\n";
  return status.code;
}

}  // namespace

int main() {
  int status_code = require_ok(orbit_wars_cuda_status(), "status");
  if (status_code != 0) {
    return status_code;
  }

  std::vector<float> input(kGameCount * kRowCount * kFeatureCount);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index % 11) - 5) / 10.0f;
  }

  std::vector<float> weights(weight_count());
  for (size_t index = 0; index < weights.size(); ++index) {
    weights[index] = std::sin(static_cast<float>(index + 1) * 0.17f) * 0.25f;
  }

  std::vector<float> output(kGameCount * kRowCount * kFeatureCount);
  OrbitWarsCudaTransformerShape shape{kLayerCount, kDModel, kHeadCount, kRowCount};
  status_code =
      require_ok(orbit_wars_cuda_trainable_forward(input.data(), weights.data(), output.data(), kGameCount, shape),
                 "trainable_forward");
  if (status_code != 0) {
    return status_code;
  }

  for (float value : output) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
      std::cerr << "operation=output_check; value=" << value << "\n";
      return 3;
    }
  }

  std::cout << "status=ok; cuda_trainable_forward_smoke=true; games=" << kGameCount
            << "; rows=" << kRowCount << "; d_model=" << kDModel << "\n";
  return 0;
}
