# Building MXBM

MXBM is a C++17 project built with CMake. It has intentionally few
dependencies.

## Dependencies

- A C++17 compiler (GCC ≥ 9, Clang ≥ 10, or Apple Clang)
- CMake ≥ 3.16
- OpenSSL (system library — for TLS pool connections)

Everything else is vendored in `third_party/` (a single-header JSON library, and
public-domain BLAKE2 and SHA-256 reference implementations).

### Linux (Arch / Omarchy)

```sh
sudo pacman -S --needed base-devel cmake openssl
```

### Debian / Ubuntu

```sh
sudo apt install build-essential cmake libssl-dev
```

### macOS

```sh
brew install cmake openssl@3
```

For the **Metal backend** (Apple Silicon), also install the Metal toolchain — it
ships separately from Xcode:

```sh
xcodebuild -downloadComponent MetalToolchain
```

This is optional. Without it CMake reports `Metal solver disabled` and every
other target still builds and tests; with it you get `--solver metal`, which is
roughly 5× faster than the OpenCL path on Apple Silicon (see
[performance.md](performance.md)).

If CMake does not find Homebrew's OpenSSL automatically, point it at the keg:

```sh
cmake -S . -B build -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

The miner binary is produced at `build/mxbm`.

Useful build types: `Release` (fastest), `RelWithDebInfo` (fast, with symbols —
recommended during development), `Debug`.

## Run the tests

MXBM ships a self-contained test suite (CTest) covering the proof-of-work
primitives, the stratum wire format, the difficulty filter, the mining engine,
the CLI/config parsing, and the HTTP API.

```sh
ctest --test-dir build --output-on-failure
```

### Optional: the Beam differential oracle

A subset of the proof-of-work tests can cross-check MXBM's verifier against
Beam's own reference implementation. This is optional and off by default. To
enable it, point the build at a local Beam checkout:

```sh
cmake -S . -B build -DBEAM_SOURCE_DIR=/path/to/beam
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Without `BEAM_SOURCE_DIR`, those tests are skipped and everything else still
builds and runs. The oracle is a build-time correctness aid only — it is never
part of a shipped binary.

## Version string

`mxbm --version` prints `MXBM v<major>.<minor>.<commit-count> [<short-hash>]`,
stamped from git at build time. Source tarballs without git metadata report
`[nogit]` for the hash.
