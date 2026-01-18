# Learn CUDA

A small, CMake-driven collection of CUDA examples. Each `.cu` file under `src/` builds into its own executable, preserving the `src/` folder structure under `build-*/bin/`.

## What’s inside

- `src/001-basic`: launch basics (streams/events, graph launch, dynamic parallelism, etc.)
- `src/002-device`: querying device properties
- `src/003-memory`: allocation and host↔device copies, shared memory
- `src/004-vector`: vector add/copy/dot/sum
- `src/005-matrix`: matrix multiply/transpose/mat-vec, WMMA example
- `src/006-ptx`: PTX-related examples

Build outputs land here (example):

- `build-release/bin/001-basic/hello-world`
- `build-debug/bin/004-vector/vector-add`

## Requirements

### For both Dev Container and local

- NVIDIA GPU + compatible NVIDIA driver
- CUDA Toolkit (provides `nvcc`)
- CMake >= 3.25
- Ninja
- A C++20-capable compiler toolchain

### Extra for VS Code Dev Container

- VS Code + “Dev Containers” extension
- Docker
- NVIDIA Container Toolkit (so the container can access your GPU)

## Setup

### Clone Dependency 

This repo expects the dependency folder to exist at `cpp-bench-utils/` (repo root).

```bash
git clone https://github.com/KindRoach/cpp-bench-utils.git
```

### a) VS Code Dev Container

1. Open this folder in VS Code.
2. Run: **Dev Containers: Reopen in Container**.
3. Ensure GPU access works inside the container (you should be able to run `nvidia-smi`).

### b) Local

1. Install CUDA Toolkit, CMake (>= 3.25), Ninja, and a C++ toolchain.
2. Clone this repo.

## How to build, run and debug

### Cmake Configure

This repo uses CMake configure presets from `CMakePresets.json`:

```bash
cmake --preset release
```

Other useful presets:

- `cmake --preset debug`
- `cmake --preset relwithdebinfo`

### Build

```bash
cmake --build build-release
```

For debug builds:

```bash
cmake --build build-debug
```

### Run

Executables are written under `build-*/bin/<category>/`.

Examples:

```bash
./build-release/bin/001-basic/hello-world
./build-release/bin/004-vector/vector-add
```

### Debug

#### VS Code (recommended)

Install the “CMake Tools” extension, then:

1. Select a configure preset (e.g. `debug` or `relwithdebinfo`).
2. Configure + build.
3. Set the target executable as the debug program (from `build-*/bin/...`).

If you use `relwithdebinfo`, you get a good balance of debug symbols and runtime speed.
