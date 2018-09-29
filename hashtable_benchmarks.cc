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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <hash_set>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "build_info.h"
#include "benchmark/benchmark.h"
#include "absl/strings/str_format.h"
#include "absl/base/port.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "boost/preprocessor.hpp"
#include "google/dense_hash_set"
#include "folly/container/F14Set.h"

#if defined(__GNUC__) || defined(__clang__)
#define HT_BENCH_FLATTEN __attribute__((__flatten__))
#define HT_BENCH_NOINLINE __attribute__((__noinline__))
#else
#define HT_BENCH_FLATTEN
#define HT_BENCH_NOINLINE
#endif

// Benchmarks for comparing hash tables.
//
// TL;DR:
//
//   $ bazel build -c opt :hashtable_benchmarks
//   $ ./bazel-bin/hashtable_benchmarks --benchmark_format=json
//
// This benchmark runs hashtables through a large suite of configurations.

namespace {

#if !defined(__clang__) && __GNUC__ == 5
// gcc 5.5 miscompiles LookupHit_Hot's DoNotOptimize(set), resulting in
// an immediate segfault in opt mode.  This fallback implementation only
// handles integral non-const lvalue types, but is sufficient for the
// hash table benchmarks.  gcc 7 is okay.
template <typename T>
std::enable_if_t<std::is_integral<T>::value || std::is_pointer<T>::value>
DoNotOptimize(T &val) {
  asm volatile("" : "+r"(val) : : "memory");
}

template <typename T>
void DoNotOptimize(T const& val) {
  asm volatile("" : : "r,m"(val) : "memory");
}
#else
using ::benchmark::DoNotOptimize;
#endif

std::mt19937 MakeRNG() {
  std::random_device rd;
  return std::mt19937(rd());
}

std::mt19937& GetRNG() {
  static auto* rng = new auto(MakeRNG());
  return *rng;
}

template <size_t kSize>
class Ballast {
  char ballast_[kSize];
};

template <>
class Ballast<0> {};

// sizeof(Value<kSize>) == kSize
// alignof(Value<kSize>) == kSize < 8 ? 4 : 8
template <size_t kSize>
class alignas(kSize < 8 ? 4 : 8) Value : private Ballast<kSize - 4> {
 public:
  static_assert(kSize >= 4, "");
  Value(uint32_t value = 0) : value_(value) {}  // NOLINT
  operator uint32_t() const { return value_; }  // NOLINT

 private:
  uint32_t value_;
};

// Use a ~zero cost hash function. The purpose of this benchmark is to focus on
// the implementations of the containers, not the quality or speed of their hash
// functions.  Bit mixing is disabled for folly::F14*.
struct Hash {
  using folly_is_avalanching = std::true_type;

  size_t operator()(uint32_t x) const noexcept {
    return (size_t{x} << 32) | x;
  }
};

struct Eq {
  bool operator()(uint32_t x, uint32_t y) const {
    ++num_calls;
    return x == y;
  }
  static size_t num_calls;
};

size_t Eq::num_calls;

// The highest order bit is set <=> it's a special value.
constexpr uint32_t kEmpty = 1U << 31;
constexpr uint32_t kDeleted = 3U << 30;

uint32_t RandomNonSpecial() {
  std::uniform_int_distribution<uint32_t> dis(0, (1U << 31) - 1);
  return dis(GetRNG());
}

// The value isn't special and the second highest order bit is unset <=>
// the value never exists in the set.
uint32_t RandomExistent() { return RandomNonSpecial() | (1U << 30); }
uint32_t RandomNonexistent() { return RandomNonSpecial() & ~(1U << 30); }

template <class Container>
void Init(Container* c) {}

template <class Container>
void Reserve(Container* c, size_t n) {
  return c->reserve(n);
}

template <class V, class H, class E>
void Init(google::dense_hash_set<V, H, E>* s) {
  s->set_empty_key(kEmpty);
  s->set_deleted_key(kDeleted);
}

template <class V, class H, class E>
void Reserve(google::dense_hash_set<V, H, E>* s, size_t n) {
  s->resize(n);
}

template <class V, class H, class E>
void Reserve(__gnu_cxx::hash_set<V, H, E>* c, size_t n) {}

template <class Container>
double MaxLoadFactor(const Container& c) {
  return c.max_load_factor();
}

template <class V, class H, class E>
double MaxLoadFactor(const __gnu_cxx::hash_set<V, H, E>& c) {
  return 1.;
}

template <class Container>
double LoadFactor(const Container& c) {
  // Do not use load_factor() because hash_map does not provide such function.
  return 1. * c.size() / c.bucket_count();
}

template <class Container>
double RelativeLoadFactor(const Container& c) {
  return LoadFactor(c) / MaxLoadFactor(c);
}

enum class Density {
  kMin,  // mininum load factor
  kMax,  // maximum load factor
};

// Returns a set filled with random data with size at least min_size and either
// low or high load factor depending on the requested density. Calling this
// function multiple times with the same arguments will yield sets with the same
// size and the same number of buckets. Their elements can be different.
template <class Set>
Set GenerateSet(size_t min_size, Density density) {
  Set set;
  Init(&set);
  Reserve(&set, min_size - 1);  // -1 is a workaround for dense_hash_set
  std::vector<uint32_t> v;
  v.reserve(min_size);
  // +1 is to ensure the final set size is at least min_size.
  while (set.size() < (density == Density::kMax ? min_size + 1 : min_size)) {
    uint32_t elem = RandomExistent();
    if (set.insert(elem).second) v.push_back(elem);
  }
  size_t bucket_count = set.bucket_count();
  while (true) {
    uint32_t elem = RandomExistent();
    if (!set.insert(elem).second) continue;
    v.push_back(elem);
    if (set.bucket_count() > bucket_count) {
      if (density == Density::kMax) {
        Set empty;
        Init(&empty);
        set.swap(empty);
        // Drop two elements instead of one as a workaround for dense_hash_set.
        assert(v.size() >= 2);
        set.insert(v.begin(), v.end() - 2);
      }
      return set;
    }
  }
}

// Generates several random sets with GenerateSet<Set>(min_size, density). The
// sum of the set sizes is at least min_total_size.
template <class Set>
HT_BENCH_NOINLINE std::vector<Set> GenerateSets(size_t min_size,
                                                size_t min_total_size,
                                                Density density) {
  GetRNG() = MakeRNG();
  size_t total_size = 0;
  std::vector<Set> res;
  res.reserve(min_total_size / min_size + 1);
  while (total_size < min_total_size) {
    Set set = GenerateSet<Set>(min_size, density);
    total_size += set.size();
    res.push_back(std::move(set));
    // This requirement makes benchmarks a bit simpler but it can be removed
    // when it becomes necessary to benchmark classes that violate it.
    assert(res.front().size() == res.back().size());
  }
  return res;
}

template <class Set>
std::vector<uint32_t> ToVector(const Set& set) {
  std::vector<uint32_t> res(set.size());
  std::copy(set.begin(), set.end(), res.begin());
  return res;
}

template <template <class...> class SetT, size_t kValueSizeT, Density kDensityT>
class Environment {
 public:
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;
  static constexpr size_t kValueSize = kValueSizeT;
  static constexpr Density kDensity = kDensityT;

  explicit Environment(benchmark::State* state) : state_(*state) {}

  size_t Size() const { return state_.range(0); }

  bool KeepRunningBatch(size_t n) {
    if (state_.iterations() == 0) Eq::num_calls = 0;
    return state_.KeepRunningBatch(n);
  }

 private:
  benchmark::State& state_;
};

template <template <class...> class SetT, size_t kValueSizeT, Density kDensityT>
const size_t Environment<SetT, kValueSizeT, kDensityT>::kValueSize;

template <template <class...> class SetT, size_t kValueSizeT, Density kDensityT>
const Density Environment<SetT, kValueSizeT, kDensityT>::kDensity;

template <class T>
std::vector<T> MakeVector(std::vector<T> v) {
  return std::move(v);
}

template <class T>
std::vector<T> MakeVector(T x) {
  std::vector<T> v;
  v.push_back(std::move(x));
  return v;
}

// Wrapper around a concrete benchmark. F must be a default-constructible
// functor taking Environment<SetT, kValueSizeT, kDensityT>* as argument and
// returning either std::vector<SetT<Value<kValueSizeT>, Hash, Eq>> or a single
// set.
template <class F, template <class...> class SetT, size_t kValueSizeT,
          Density kDensityT>
void BM(benchmark::State& state) {
  // Reset RNG so that each benchmark sees the same sequence of random numbers.
  GetRNG() = MakeRNG();
  Environment<SetT, kValueSizeT, kDensityT> env(&state);
  std::vector<SetT<Value<kValueSizeT>, Hash, Eq>> s = MakeVector(F()(&env));
  // No iterations means the benchmark is disabled.
  if (state.iterations() == 0) return;
  assert(!s.empty());
  assert(!s.front().empty());
  for (const auto& set : s) {
    assert(set.size() == s.front().size());
    assert(set.bucket_count() == s.front().bucket_count());
    (void)set; // silence warning in opt
  }
  if (kDensityT == Density::kMin) {
    assert(RelativeLoadFactor(s.front()) < 0.6);
  } else {
    assert(RelativeLoadFactor(s.front()) > 0.6);
  }
  assert(state.iterations() >  0);
  double comp_per_op = 1. * Eq::num_calls / state.iterations();
  state.SetLabel(absl::StrFormat("lf=%.2f cmp=%.3f size=%u num_sets=%u",
                                 LoadFactor(s.front()), comp_per_op,
                                 s.front().size(), s.size()));
}

// Transposes the matrix and concatenates the resulting elements.
template <class T>
std::vector<T> Transpose(std::vector<std::vector<T>> m) {
  assert(!m.empty());
  std::vector<T> v(m.size() * m.front().size());
  for (size_t i = 0; i != m.size(); ++i) {
    assert(m[i].size() == m[0].size());
    for (size_t j = 0; j != m[i].size(); ++j) {
      v[j * m.size() + i] = m[i][j];
    }
  }
  return v;
}

// Helper function used to implement two similar benchmarks defined below.
template <class Env, class Lookup>
HT_BENCH_FLATTEN std::vector<typename Env::Set> LookupHit_Hot(Env* env,
                                                              Lookup lookup) {
  using Set = typename Env::Set;
  static constexpr size_t kMinTotalKeyCount = 64 << 10;
  static constexpr size_t kOpsPerKey = 512;

  std::vector<Set> s =
      GenerateSets<Set>(env->Size(), kMinTotalKeyCount, Env::kDensity);

  if (s.size() > 1) {
    while (env->KeepRunningBatch(s.size() * s.front().size() * kOpsPerKey)) {
      for (Set& set : s) {
        for (uint32_t key : set) {
          for (size_t i = 0; i != kOpsPerKey; ++i) {
            lookup(&set, key);
          }
        }
      }
    }
  } else {
    std::vector<uint32_t> keys = ToVector(s.front());
    std::shuffle(keys.begin(), keys.end(), GetRNG());
    keys.resize(kMinTotalKeyCount);
    while (env->KeepRunningBatch(kMinTotalKeyCount * kOpsPerKey)) {
      for (uint32_t key : keys) {
        for (size_t i = 0; i != kOpsPerKey; ++i) {
          lookup(&s.front(), key);
        }
      }
    }
  }

  return s;
}

// Helper function used to implement two similar benchmarks defined below.
template <class Env, class Lookup>
HT_BENCH_FLATTEN std::vector<typename Env::Set> LookupHit_Cold(Env* env,
                                                               Lookup lookup) {
  using Set = typename Env::Set;
  // The larger this value, the colder the benchmark and the longer it takes to
  // run.
  static constexpr size_t kMinTotalBytes = 256 << 20;

  std::vector<Set> s = GenerateSets<Set>(
      env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);

  std::vector<std::vector<uint32_t>> m(s.size());
  for (size_t i = 0; i != s.size(); ++i) {
    m[i] = ToVector(s[i]);
    std::shuffle(m[i].begin(), m[i].end(), GetRNG());
  }
  std::vector<uint32_t> keys = Transpose(std::move(m));

  const size_t num_sets = s.front().size();
  const size_t set_size = s.size();
  while (env->KeepRunningBatch(keys.size())) {
    for (size_t i = 0; i != num_sets; ++i) {
      for (size_t j = 0; j != set_size; ++j) {
        lookup(&s[j], keys[i * set_size + j]);
      }
    }
  }

  return s;
}

// Measures the time it takes to `find` an existent element.
//
//   assert(set.find(key) != set.end());
struct FindHit_Hot {
  template <class Env>
  std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    return LookupHit_Hot(env, [](Set* set, uint32_t key) {
      DoNotOptimize(set);
      DoNotOptimize(key);
      DoNotOptimize(set->find(key));
    });
  }
};

// Measures the time it takes to `find` an existent element.
//
//   assert(set.find(key) != set.end());
struct FindHit_Cold {
  template <class Env>
  std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    return LookupHit_Cold(env, [](Set* set, uint32_t key) {
      DoNotOptimize(set);
      DoNotOptimize(key);
      DoNotOptimize(set->find(key));
    });
  }
};

// Measures the time it takes to `insert` an existent element.
//
//   assert(!set.insert(key).second);
struct InsertHit_Hot {
  template <class Env>
  std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    return LookupHit_Hot(env, [](Set* set, uint32_t key) {
      DoNotOptimize(set);
      DoNotOptimize(key);
      DoNotOptimize(set->insert(key));
    });
  }
};

// Measures the time it takes to `insert` an existent element.
//
//   assert(!set.insert(key).second);
struct InsertHit_Cold {
  template <class Env>
  std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    return LookupHit_Cold(env, [](Set* set, uint32_t key) {
      DoNotOptimize(set);
      DoNotOptimize(key);
      DoNotOptimize(set->insert(key));
    });
  }
};

// Measures the time it takes to `find` an nonexistent element.
//
//   assert(set.find(key) == set.end());
struct FindMiss_Hot {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the less the results will depend on randomness and
    // the longer the benchmark will run.
    static constexpr size_t kMinTotalKeyCount = 64 << 10;
    // The larger this value, the hotter the benchmark and the longer it will
    // run.
    static constexpr size_t kOpsPerKey = 512;

    std::vector<Set> s =
        GenerateSets<Set>(env->Size(), kMinTotalKeyCount, Env::kDensity);
    const size_t keys_per_set = kMinTotalKeyCount / s.size();

    while (env->KeepRunningBatch(s.size() * keys_per_set * kOpsPerKey)) {
      for (const Set& set : s) {
        for (size_t i = 0; i != keys_per_set; ++i) {
          const uint32_t key = RandomNonexistent();
          for (size_t j = 0; j != kOpsPerKey; ++j) {
            DoNotOptimize(set);
            DoNotOptimize(key);
            DoNotOptimize(set.find(key));
          }
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `find` an nonexistent element.
//
//   assert(set.find(key) == set.end());
struct FindMiss_Cold {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the colder the benchmark and the longer it takes
    // to run.
    static constexpr size_t kMinTotalBytes = 256 << 20;

    std::vector<Set> s = GenerateSets<Set>(
        env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);
    std::vector<uint32_t> keys(s.front().size());
    for (uint32_t& key : keys) key = RandomNonexistent();

    while (env->KeepRunningBatch(keys.size() * s.size())) {
      for (uint32_t key : keys) {
        for (const Set& set : s) {
          DoNotOptimize(set);
          DoNotOptimize(key);
          DoNotOptimize(set.find(key));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `erase` an existent element and then `insert`
// a new element.
//
//   assert(set.erase(key1));
//   assert(set.insert(key2).second);
struct EraseInsert_Hot {
  template <class Env>
  HT_BENCH_FLATTEN typename Env::Set operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the less the results will depend on randomness and
    // the longer the benchmark will run.
    static constexpr size_t kMinKeyCount = 1 << 20;

    // I cannot figure out how to make this benchmark work with high load factor
    // on dense_hash_set and flat_hash_set. Disabling it for now.
    if (Env::kDensity != Density::kMin) return Set();

    Set s = GenerateSet<Set>(env->Size(), Env::kDensity);
    std::vector<uint32_t> keys = ToVector(s);
    std::shuffle(keys.begin(), keys.end(), GetRNG());
    std::unordered_set<uint32_t> extra;
    while (keys.size() < kMinKeyCount || keys.size() < 3 * s.size()) {
      uint32_t key = RandomExistent();
      if (!s.count(key) && extra.insert(key).second) keys.push_back(key);
    }

    const size_t set_size = s.size();
    const size_t num_keys = keys.size();
    while (env->KeepRunningBatch(num_keys)) {
      for (size_t i = 0; i != num_keys - set_size; ++i) {
        DoNotOptimize(s);
        DoNotOptimize(s.erase(keys[i]));
        DoNotOptimize(s.insert(keys[i + set_size]));
      }
      for (size_t i = 0; i != set_size; ++i) {
        DoNotOptimize(s);
        DoNotOptimize(s.erase(keys[num_keys - set_size + i]));
        DoNotOptimize(s.insert(keys[i]));
      }
    }

    return s;
  }
};

// Measures the time it takes to `erase` an existent element and then `insert`
// a new element.
//
//   assert(set.erase(key1));
//   assert(set.insert(key2).second);
struct EraseInsert_Cold {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the colder the benchmark and the longer it takes
    // to run.
    static constexpr size_t kMinTotalBytes = 128 << 20;

    // I cannot figure out how to make this benchmark work with high load factor
    // on dense_hash_set and flat_hash_set. Disabling it for now.
    if (Env::kDensity != Density::kMin) return {};

    std::vector<Set> s = GenerateSets<Set>(
        env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);

    const size_t set_size = s.front().size();
    const size_t set_keys = 3 * set_size;
    std::vector<std::vector<uint32_t>> m(s.size());
    for (size_t i = 0; i != s.size(); ++i) {
      m[i] = ToVector(s[i]);
      std::shuffle(m[i].begin(), m[i].end(), GetRNG());
      std::unordered_set<uint32_t> extra;
      m[i].reserve(set_keys);
      while (m[i].size() < set_keys) {
        uint32_t key = RandomExistent();
        if (!s[i].count(key) && extra.insert(key).second) m[i].push_back(key);
      }
    }

    std::vector<uint32_t> keys = Transpose(std::move(m));

    const size_t num_sets = s.size();
    while (env->KeepRunningBatch(keys.size())) {
      for (size_t i = 0; i != set_keys - set_size; ++i) {
        for (size_t j = 0; j != num_sets; ++j) {
          DoNotOptimize(s[j].erase(keys[i * num_sets + j]));
          DoNotOptimize(s[j].insert(keys[(i + set_size) * num_sets + j]));
        }
      }
      for (size_t i = 0; i != set_size; ++i) {
        for (size_t j = 0; j != num_sets; ++j) {
          DoNotOptimize(
              s[j].erase(keys[(set_keys - set_size + i) * num_sets + j]));
          DoNotOptimize(s[j].insert(keys[i * num_sets + j]));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `clear` a set and then `insert` the same
// elements back in random order. The reported time is per element. In other
// words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.reserve(N);
//   set.insert(key1);
//   ...
//   set.insert(keyN);
//
// What we really need is to clear the container without releasing memory. For
// most containers this can be expressed as `set.clear()` but for SwissTable
// and densehashtable containers this call can release memory.
struct InsertManyUnordered_Hot {
  template <class Env>
  HT_BENCH_FLATTEN typename Env::Set operator()(Env* env) const {
    using Set = typename Env::Set;
    // The higher the value, the less contribution std::shuffle makes. The price
    // is longer benchmarking time. With 64 std::shuffle adds around 0.3 ns to
    // the benchmark results.
    static constexpr size_t kRepetitions = 64;
    // The larger this value, the less the results will depend on randomness and
    // the longer the benchmark will run.
    static constexpr size_t kMinInsertions = 256 << 10;

    Set s = GenerateSet<Set>(env->Size(), Env::kDensity);
    std::vector<uint32_t> keys = ToVector(s);

    const size_t n = std::max(size_t{1}, kMinInsertions / keys.size());
    while (env->KeepRunningBatch(keys.size() * n * kRepetitions)) {
      for (size_t i = 0; i != n; ++i) {
        std::shuffle(keys.begin(), keys.end(), GetRNG());
        for (size_t j = 0; j != kRepetitions; ++j) {
          s.clear();
          Reserve(&s, keys.size());
          for (uint32_t key : keys) DoNotOptimize(s.insert(key));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `clear` a set and then `insert` the same
// elements back in random order. The reported time is per element. In other
// words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.reserve(N);
//   set.insert(key1);
//   ...
//   set.insert(keyN);
//
// What we really need is to clear the container without releasing memory. For
// most containers this can be expressed as `set.clear()` but for SwissTable
// and densehashtable containers this call can release memory.
struct InsertManyUnordered_Cold {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the colder the benchmark and the longer it takes
    // to run.
    static constexpr size_t kMinTotalBytes = 128 << 20;

    std::vector<Set> s = GenerateSets<Set>(
        env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);

    std::vector<std::vector<uint32_t>> m(s.size());
    for (size_t i = 0; i != s.size(); ++i) {
      m[i] = ToVector(s[i]);
      std::shuffle(m[i].begin(), m[i].end(), GetRNG());
    }

    std::vector<uint32_t> keys = Transpose(std::move(m));
    const size_t set_size = s.front().size();
    const size_t num_sets = s.size();
    while (env->KeepRunningBatch(keys.size())) {
      for (Set& set : s) {
        set.clear();
        Reserve(&set, set_size);
      }
      for (size_t i = 0; i != set_size; ++i) {
        for (size_t j = 0; j != num_sets; ++j) {
          DoNotOptimize(s[j].insert(keys[i * num_sets + j]));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `clear` a set and then `insert` the same
// elements in the order they were in the set. The reported time is per element.
// In other words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.reserve(N);
//   set.insert(key1);
//   ...
//   set.insert(keyN);
//
// What we really need is to clear the container without releasing memory. For
// most containers this can be expressed as `set.clear()` but for SwissTable,
// densehashtable, and F14 containers this call can release memory.
struct InsertManyOrdered_Hot {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The higher the value, the less contribution std::shuffle makes. The price
    // is longer benchmarking time. With 64 std::shuffle adds around 0.3 ns to
    // the benchmark results.
    static constexpr size_t kRepetitions = 64;
    // The larger this value, the less the results will depend on randomness and
    // the longer the benchmark will run.
    static constexpr size_t kMinTotalKeyCount = 256 << 10;

    std::vector<Set> s =
        GenerateSets<Set>(env->Size(), kMinTotalKeyCount, Env::kDensity);
    std::vector<std::vector<uint32_t>> keys(s.size());
    for (size_t i = 0; i != s.size(); ++i) {
      keys[i] = ToVector(s[i]);
    }

    while (env->KeepRunningBatch(s.size() * s.front().size() * kRepetitions)) {
      for (size_t i = 0; i != s.size(); ++i) {
        for (size_t j = 0; j != kRepetitions; ++j) {
          s[i].clear();
          Reserve(&s[i], keys[i].size());
          for (uint32_t key : keys[i]) DoNotOptimize(s[i].insert(key));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to `clear` a set and then `insert` the same
// elements in the order they were in the set. The reported time is per element.
// In other words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.reserve(N);
//   set.insert(key1);
//   ...
//   set.insert(keyN);
//
// What we really need is to clear the container without releasing memory. For
// most containers this can be expressed as `set.clear()` but for SwissTable
// and densehashtable containers this call can release memory.
struct InsertManyOrdered_Cold {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the colder the benchmark and the longer it takes
    // to run.
    static constexpr size_t kMinTotalBytes = 128 << 20;

    std::vector<Set> s = GenerateSets<Set>(
        env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);

    std::vector<std::vector<uint32_t>> m(s.size());
    for (size_t i = 0; i != s.size(); ++i) {
      m[i] = ToVector(s[i]);
    }

    std::vector<uint32_t> keys = Transpose(std::move(m));
    const size_t set_size = s.front().size();
    const size_t num_sets = s.size();
    while (env->KeepRunningBatch(keys.size())) {
      for (Set& set : s) {
        set.clear();
        Reserve(&set, set_size);
      }
      for (size_t i = 0; i != set_size; ++i) {
        for (size_t j = 0; j != num_sets; ++j) {
          DoNotOptimize(s[j].insert(keys[i * num_sets + j]));
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to iterate over a set and read its every element.
// The reported time is per element. In other words, the pseudo code below
// counts as `set.size()` iterations.
//
//   for (const auto& elem : set) {
//     Read(elem);
//   }
struct Iterate_Hot {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the hotter the benchmark and the longer it will
    // run.
    static constexpr size_t kRepetitions = 64;
    // The larger this value, the less the results will depend on randomness and
    // the longer the benchmark will run.
    static constexpr size_t kMinKeyCount = 256 << 10;

    std::vector<Set> s =
        GenerateSets<Set>(env->Size(), kMinKeyCount, Env::kDensity);

    alignas(Value<Env::kValueSize>) char data[Env::kValueSize];
    while (env->KeepRunningBatch(s.size() * s.front().size() * kRepetitions)) {
      for (const Set& set : s) {
        for (size_t i = 0; i != kRepetitions; ++i) {
          for (const auto& elem : set) {
            memcpy(data, &elem, Env::kValueSize);
            DoNotOptimize(data);
          }
        }
      }
    }

    return s;
  }
};

// Measures the time it takes to iterate over a set and read its every element.
// The reported time is per element. In other words, the pseudo code below
// counts as `set.size()` iterations.
//
//   for (const auto& elem : set) {
//     Read(elem);
//   }
struct Iterate_Cold {
  template <class Env>
  HT_BENCH_FLATTEN std::vector<typename Env::Set> operator()(Env* env) const {
    using Set = typename Env::Set;
    // The larger this value, the colder the benchmark and the longer it takes
    // to run.
    static constexpr size_t kMinTotalBytes = 128 << 20;
    static constexpr size_t kStride = 16;

    std::vector<Set> s = GenerateSets<Set>(
        env->Size(), kMinTotalBytes / Env::kValueSize, Env::kDensity);

    const size_t num_strides = s.front().size() / kStride;
    assert(num_strides > 0);

    std::vector<std::vector<typename Set::const_iterator>> m;
    m.reserve(s.size());
    for (const Set& set : s) {
      auto iter = set.begin();
      std::vector<typename Set::const_iterator> iters;
      iters.reserve(num_strides);
      for (size_t i = 0; i != num_strides; ++i) {
        iters.push_back(iter);
        std::advance(iter, kStride);
      }
      m.push_back(std::move(iters));
    }

    std::vector<typename Set::const_iterator> begins = Transpose(std::move(m));
    std::vector<typename Set::const_iterator> iters = begins;
    alignas(Value<Env::kValueSize>) char data[Env::kValueSize];
    const size_t num_sets = s.size();
    while (env->KeepRunningBatch(iters.size() * kStride)) {
      for (size_t i = 0; i != kStride; ++i) {
        for (size_t j = 0; j != num_strides; ++j) {
          for (size_t k = 0; k != num_sets; ++k) {
            auto& iter = iters[j * num_sets + k];
            DoNotOptimize(iter == s[k].end());
            memcpy(data, &*iter, Env::kValueSize);
            DoNotOptimize(data);
            ++iter;
          }
        }
      }
      iters = begins;
    }

    return s;
  }
};


// Adjust benchmark flag defaults so that the benchmarks run in a reasonable
// time. The benchmark framework restarts the whole benchmark with number of
// iterations until it runs for a specific time. This means we build the
// benchmarking state many times over and this is very costly. To fix this
// we set a fixed high number iterations that experimentally produce stable
// results and avoid the repeated creation of the benchmarking state.
void ConfigureBenchmark(benchmark::internal::Benchmark* b) {
  b->Arg(1 << 4);
  b->Arg(1 << 8);
  b->Arg(1 << 12);
  b->Arg(1 << 16);
  b->Arg(1 << 20);
  b->Iterations(20000000);
}

// clang-format off
#define BENCHES         \
  (FindHit)             \
  (FindMiss)            \
  (InsertHit)           \
  (EraseInsert)         \
  (InsertManyUnordered) \
  (InsertManyOrdered)   \
  (Iterate)

#define ENVS \
    (_Hot)   \
    (_Cold)

#define VALUE_SIZES \
    (4)             \
    (64)

#define DENSITIES   \
    (Density::kMin) \
    (Density::kMax)

#define SET_TYPES          \
  (__gnu_cxx::hash_set)    \
  (std::unordered_set)     \
  (absl::flat_hash_set)    \
  (absl::node_hash_set)    \
  (folly::F14ValueSet)     \
  (folly::F14NodeSet)      \
  (folly::F14VectorSet)    \
  (google::dense_hash_set)
// clang-format on

#define DEFINE_BENCH_I(bench, env, set, value_size, density)   \
  BENCHMARK_TEMPLATE(BM, bench##env, set, value_size, density) \
      ->Apply(ConfigureBenchmark);

#define DEFINE_BENCH(r, seq) \
  BOOST_PP_EXPAND(DEFINE_BENCH_I BOOST_PP_SEQ_TO_TUPLE(seq))

BOOST_PP_SEQ_FOR_EACH_PRODUCT(
    DEFINE_BENCH, (BENCHES)(ENVS)(SET_TYPES)(VALUE_SIZES)(DENSITIES))

}  // namespace

int main(int argc, char **argv) {
  auto prog = std::string(argv[0]) + "@" + getBuildScmRevision();
  argv[0] = const_cast<char*>(prog.c_str());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
