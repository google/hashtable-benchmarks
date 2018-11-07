// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "bytell_hash_map.hpp"

// ska::bytell_hash_set has a collision-based rehash decision.
// This program explores the load factor at which that rehash is triggered.

int main(int argc, char **argv) {
  size_t maxSize = 1U << 28;
  size_t numReps = 100;

  std::random_device rd;
  auto rng = std::mt19937{rd()};
  auto dis = std::uniform_int_distribution<uint32_t>{0, (1U << 31) - 1};

  for (size_t rep = 0; rep < numReps; ++rep) {
    ska::bytell_hash_set<uint32_t> set;
    set.max_load_factor(0.999);

    while (set.size() < maxSize) {
      auto key = dis(rng);
      size_t prevSize = set.size();
      size_t prevCap = set.bucket_count();
      set.insert(key);
      if (set.bucket_count() > prevCap && prevCap > 0) {
        auto lf = static_cast<double>(prevSize) / prevCap;
        std::cout << prevCap << " " << prevSize << " " << lf << "\n";
      }
    }
    std::cout << "\n";
  }
  return 0;
}
