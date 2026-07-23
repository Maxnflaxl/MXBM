# Contributing

Contributions are welcome. This document covers the workflow and the one rule
that is specific to MXBM: clean-room provenance.

## Workflow

1. Build and run the tests (see [building.md](building.md)):
   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```
2. Make your change with a focused test that fails before and passes after.
3. Keep the full suite green and the build warning-clean
   (`-Wall -Wextra -Wpedantic`).
4. Open a pull request describing what changed and why.

## Code style

Follow the surrounding code. In short: C++17, small focused files each with one
clear responsibility, standard-library-only in the shipping paths (no Boost),
and no new third-party dependencies without discussion. New proof-of-work or
wire-format code should come with golden vectors, not just round-trip tests.

## Clean-room provenance

MXBM is a clean-room implementation. **Do not copy code, disassembly, or
extracted data from any closed-source miner** into this repository — not as
source, not as vendored blobs, not in tests. MXBM's behavior may *resemble*
other miners (console output shapes, command-line conventions, configuration
file formats are functional facts, freely reimplemented), but its expression is
original.

Code derived from **Beam** is fine and encouraged: Beam is Apache-2.0 licensed,
the same license as MXBM. When you port from Beam, keep the attribution comment
naming the source file, and make sure the dependency is recorded in
[NOTICE](../NOTICE).

## Licensing of contributions

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](../LICENSE), the same terms as the project.
