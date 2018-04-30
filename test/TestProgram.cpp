#include <iostream>
#include <string>
#include <vector>
#include "Kernel.h"

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./TestProgram <N>" << std::endl;
    return 1;
  }
  const int n = std::stoi(argv[1]);
  std::vector<Data_t> x(n, 4);
  std::vector<Data_t> y(n, 4);
  std::vector<Data_t> out(1, 0);
  blas_xdot(x.data(), y.data(), out.data(), n);
  Data_t expected = 4 * 4 * n;
  if (out[0] != expected) {
    std::cerr << "Mismatch: " << out[0] << " (expected " << expected << ')' << std::endl;
    return 1;
  } else {
    std::cout << "Got correct result" << std::endl;
  }
  return 0;
}
