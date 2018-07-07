workspace(name = "com_google_hashtable_benchmarks")

# abseil
git_repository(
    name = "absl",
    commit = "8f612ebb152fb7e05643a2bcf78cb89a8c0641ad",
    remote = "https://github.com/abseil/abseil-cpp",
)

# Google benchmark.
git_repository(
    name = "google_benchmark",
    commit = "16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
    remote = "https://github.com/google/benchmark",
)

# Google dense_hash_set
git_repository(
    name = "google_sparsehash",
    commit = "4cb924025b8c622d1a1e11f4c1e9db15410c75fb",
    remote = "https://github.com/google/sparsehash",
)

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "239ce40e42ab0e3fe7ce84c2e9303ff8a277c41a",
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()
