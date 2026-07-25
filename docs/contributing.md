# Contributing

Contributions are welcome. This document covers the workflow and the code style.

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

## Attribution

Code ported from **Beam** is fine and encouraged: Beam is Apache-2.0, the same
licence as MXBM. Keep the attribution comment naming the source file, and record
the dependency in [NOTICE](../NOTICE).

## Licensing of contributions

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](../LICENSE), the same terms as the project.
