# Copyright 2018 Google Inc.  All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
