# Bundled BaM components

This directory contains the userspace portion of BaM required by TopoANNS. It
is derived from BaM commit `d20ef4ac5d4513a411422e41caf48562977504dc`.
TopoANNS adds compact per-page cache metadata, physical-read counters,
outstanding-I/O profiling, and write-path compatibility helpers.

Only the userspace library and headers used by TopoANNS are vendored here. The
BaM kernel module and system setup instructions remain in the upstream
project:

https://github.com/ZaidQureshi/bam

The `include/freestanding` directory is a pinned Git submodule. Initialize it
with:

```bash
git submodule update --init --recursive
```

BaM is distributed under the BSD license in [LICENSE](LICENSE).
