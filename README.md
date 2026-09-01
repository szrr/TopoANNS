# TopoANNS

TopoANNS is an I/O-efficient, GPU-centric approximate nearest-neighbor search
system for graph indices stored on NVMe SSDs. Its search path combines
GPU-initiated direct I/O, topology-aware caching, PQ-based traversal and
refinement, learned traversal stopping, learned exact-I/O admission, and
cross-microbatch overlap.

This repository contains the research implementation, index preparation
utilities, evaluation binaries, and unit tests. Datasets, trained models,
generated indices, and experiment outputs are intentionally not included.

## Repository layout

- `include/topoanns`: public C++ interfaces.
- `src`: index I/O, GPU search, reranking, providers, and RVQ support.
- `tools`: index preparation, SSD loading, evaluation, and profiling tools.
- `tests`: unit and GPU integration tests.
- `config`: an example BaM runtime configuration.
- `third_party/bam`: the minimal BaM userspace dependency used by TopoANNS.

## Requirements

- Linux on x86-64.
- CMake 3.27 or newer and a C++17 compiler.
- CUDA Toolkit 12.3 or newer.
- BLAS and OpenMP development packages when building the evaluation tools.
- A BaM-compatible NVIDIA GPU/NVMe platform and the BaM kernel module for
  direct SSD access.

The checked-in configuration targets CUDA architecture 89, matching the
validated NVIDIA L40 platform. Set `TOPOANNS_CUDA_ARCHITECTURES` when building
for another supported architecture. Direct-I/O execution also requires the
hardware and system configuration described by the
[BaM project](https://github.com/ZaidQureshi/bam).

## Build

Clone with submodules, then configure and build:

```bash
git clone --recursive https://github.com/szrr/TopoANNS.git
cd TopoANNS
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTOPOANNS_CUDA_ARCHITECTURES=89
cmake --build build -j
```

If an older CUDA installation is the system default, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda-12.3/bin/nvcc` to the configure
command.

The bundled BaM userspace library is built by default. To use an existing
library, pass `-DTOPOANNS_BAM_LIBRARY=/absolute/path/to/libnvm.so`.

DiskANN-dependent PQ construction tools are disabled by default because they
require an external DiskANN/MKL build. Enable them with
`-DTOPOANNS_BUILD_DISKANN_TOOLS=ON` and set
`TOPOANNS_DISKANN_ROOT` and `TOPOANNS_DISKANN_LIBRARY`.

Useful build switches:

- `TOPOANNS_BUILD_TOOLS=ON|OFF`
- `TOPOANNS_BUILD_TESTS=ON|OFF`
- `TOPOANNS_BUILD_DISKANN_TOOLS=ON|OFF`

## Runtime configuration

Create a machine-local BaM configuration:

```bash
cp config/bam_runtime.conf.example config/bam_runtime.conf
```

Set the controller device, CUDA device, namespace, cache size, queue depth, and
queue count in that file. Programs also accept
`--bam-config-path /path/to/bam_runtime.conf`. The local configuration is
ignored by Git.

## Index preparation and search

The subset builders convert a DiskANN graph, PQ data, and raw vectors into the
TopoANNS topology and vector-store layouts:

```bash
build/topoanns_build_sift_subset \
  --base-bvecs <base.bvecs> \
  --disk-index <disk.index> \
  --pq-pivots <pq_pivots.bin> \
  --pq-compressed <pq_codes.bin> \
  --output-dir <index_dir> \
  --num-vectors <count>
```

Use `topoanns_build_float_subset` with `--base-fbin` for float-vector
datasets. The BaM write tools copy the generated topology, vector, or combined
layout to the configured SSD. Run each tool without arguments to print its
complete command-line interface.

A representative Recall/QPS sweep is:

```bash
sudo build/topoanns_eval_sift \
  --index-dir <index_dir> \
  --rvq-model <rvq_model.bin> \
  --query-bin <queries.bin> \
  --gt-bin <ground_truth.bin> \
  --num-queries 10000 \
  --top-l-values 16,32,64,128,256,512,1024 \
  --batch-size 10000 \
  --top-k 10 \
  --search-width 2 \
  --rerank-top-n 32 \
  --rerank-mode persistent \
  --max-expansions 4096 \
  --rvq-entry-count 128
```

## Tests

After building, run:

```bash
ctest --test-dir build --output-on-failure
```

GPU and BaM integration tests require a compatible configured device; the
format and host-side tests do not access an SSD.

## Safety

The `topoanns_bam_write_*` programs write directly to the raw device exposed
by the BaM driver. Verify the controller path and byte offsets before running
them. Never point these tools at a device containing data that must be
preserved.

## License

TopoANNS is licensed under Apache-2.0. Vendored BaM components retain their BSD
license in `third_party/bam/LICENSE`.
