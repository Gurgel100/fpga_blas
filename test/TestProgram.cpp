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
  std::vector<Data_t> in(n, 4);
  std::vector<Data_t> out(n, 0);
  AddOne(in.data(), out.data(), n);
  for (auto &i : out) {
    if (i != 5) {
      std::cerr << "Mismatch: " << i << " (expected 5)" << std::endl;
      return 1;
    }
  }
  return 0;
}
