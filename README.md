# Hashtable Benchmarks

Benchmarks for comparing hashtable implementations.

1. Build:

```shell
bazel build :hashtable_benchmarks
```

Note that `-c opt` is the default.

2. Run:

```shell
./bazel-bin/hashtable_benchmarks --benchmark_format=json > benchmark-results.json
```

3. Analyze:

You can use http://colab.research.google.com along with `hashtable_benchmarks.ipynb` to parse the generated `benchmark-results.json`.

4. Contribute:

We would like this to turn into *the* place for comparing hashtables in C++.  We
will accept external dependencies on other hashtable libraries (assuming they
have a compatible licence).  We encourage folks to improve and modify both the
analysis and the benchmarks themselves as we learn things.  Please join the
dicussion at

https://groups.google.com/forum/#!forum/hashtable-benchmarks

# Disclaimer

This is not an officially supported Google product.
