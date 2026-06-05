#include "orbit_wars_v8_cuda.h"

#include <iostream>

int main() {
  OrbitWarsV8CudaStatus status = orbit_wars_cuda_v8_status();
  if (status.code != 0) {
    std::cerr << "status=failed; code=" << status.code << "; message=" << status.message << "\n";
    return status.code;
  }
  std::cout << "status=ok; cuda_v8_status=true\n";
  return 0;
}
