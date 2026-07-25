cmake_minimum_required(VERSION 3.16)
project(MXBM VERSION 0.5 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
enable_testing()

add_executable(test_smoke tests/test_smoke.cpp)
target_include_directories(test_smoke PRIVATE tests src/beamhash)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_primitives tests/test_primitives.cpp)
target_include_directories(test_primitives PRIVATE tests src/beamhash)
add_test(NAME primitives COMMAND test_primitives)

include(cmake/beam_oracle.cmake)

add_library(mxbm_beamhash STATIC src/beamhash/bh3_blake2b.cpp)
target_include_directories(mxbm_beamhash PUBLIC src/beamhash)
target_link_libraries(mxbm_beamhash PUBLIC mxbm_blake2b)
target_sources(mxbm_beamhash PRIVATE src/beamhash/bh3_verify.cpp)

target_link_libraries(test_primitives PRIVATE mxbm_beamhash)

add_executable(test_verify tests/test_verify.cpp)
target_include_directories(test_verify PRIVATE tests src/beamhash)
target_link_libraries(test_verify PRIVATE mxbm_beamhash)
add_test(NAME verify COMMAND test_verify)

if(MXBM_HAVE_BEAM_ORACLE)
  add_executable(test_oracle_spike tests/test_oracle_spike.cpp)
  target_include_directories(test_oracle_spike PRIVATE tests)
  target_link_libraries(test_oracle_spike PRIVATE beam_oracle)
  add_test(NAME oracle_spike COMMAND test_oracle_spike)

  add_executable(test_differential tests/test_differential.cpp)
  target_include_directories(test_differential PRIVATE tests src/beamhash)
  target_link_libraries(test_differential PRIVATE mxbm_beamhash beam_oracle)
  add_test(NAME differential COMMAND test_differential)
endif()

find_package(OpenSSL REQUIRED)   # transport (later task); fail early if absent
add_library(mxbm_json INTERFACE)
target_include_directories(mxbm_json INTERFACE third_party)

add_executable(test_wire_spike tests/test_wire_spike.cpp)
target_include_directories(test_wire_spike PRIVATE tests src)
target_link_libraries(test_wire_spike PRIVATE mxbm_json)
add_test(NAME wire_spike COMMAND test_wire_spike)

add_library(mxbm_stratum STATIC src/stratum/messages.cpp)
target_include_directories(mxbm_stratum PUBLIC src)
target_link_libraries(mxbm_stratum PUBLIC mxbm_json)

add_executable(test_messages tests/test_messages.cpp)
target_include_directories(test_messages PRIVATE tests src)
target_link_libraries(test_messages PRIVATE mxbm_stratum)
add_test(NAME messages COMMAND test_messages)

add_library(mxbm_pow STATIC src/pow/difficulty.cpp third_party/sha256/sha256.c)
target_include_directories(mxbm_pow PUBLIC src third_party)

add_executable(test_difficulty tests/test_difficulty.cpp)
target_include_directories(test_difficulty PRIVATE tests src third_party)
target_link_libraries(test_difficulty PRIVATE mxbm_pow)
add_test(NAME difficulty COMMAND test_difficulty)

target_sources(mxbm_stratum PRIVATE src/stratum/transport.cpp)
target_link_libraries(mxbm_stratum PUBLIC OpenSSL::SSL OpenSSL::Crypto)

find_package(Threads REQUIRED)
add_executable(test_transport_loopback tests/test_transport_loopback.cpp)
target_include_directories(test_transport_loopback PRIVATE tests src)
target_link_libraries(test_transport_loopback PRIVATE mxbm_stratum Threads::Threads)
add_test(NAME transport_loopback COMMAND test_transport_loopback)

target_sources(mxbm_stratum PRIVATE src/stratum/client.cpp)
target_link_libraries(mxbm_stratum PUBLIC Threads::Threads)

add_executable(test_client tests/test_client.cpp)
target_include_directories(test_client PRIVATE tests src)
target_link_libraries(test_client PRIVATE mxbm_stratum)
add_test(NAME client COMMAND test_client)

add_library(mxbm_miner STATIC src/miner/engine.cpp src/miner/benchmark.cpp)
target_include_directories(mxbm_miner PUBLIC src)
target_link_libraries(mxbm_miner PUBLIC mxbm_stratum mxbm_pow)
target_sources(mxbm_miner PRIVATE src/miner/stats.cpp)
target_sources(mxbm_miner PRIVATE src/miner/devfee.cpp)

if(MXBM_HAVE_BEAM_ORACLE)
  target_sources(mxbm_miner PRIVATE src/miner/solver_ref.cpp)
  target_link_libraries(mxbm_miner PUBLIC beam_oracle)
endif()

add_executable(test_engine tests/test_engine.cpp)
target_include_directories(test_engine PRIVATE tests src)
target_link_libraries(test_engine PRIVATE mxbm_miner)
add_test(NAME engine COMMAND test_engine)

add_executable(test_engine_iterate tests/test_engine_iterate.cpp)
target_include_directories(test_engine_iterate PRIVATE tests src)
target_link_libraries(test_engine_iterate PRIVATE mxbm_miner Threads::Threads)
add_test(NAME engine_iterate COMMAND test_engine_iterate)

add_executable(test_stats tests/test_stats.cpp)
target_include_directories(test_stats PRIVATE tests src)
target_link_libraries(test_stats PRIVATE mxbm_miner)
add_test(NAME stats COMMAND test_stats)

add_executable(test_devfee tests/test_devfee.cpp)
target_include_directories(test_devfee PRIVATE tests src)
target_link_libraries(test_devfee PRIVATE mxbm_miner Threads::Threads)
add_test(NAME devfee COMMAND test_devfee)

add_executable(test_cli tests/test_cli.cpp src/cli/options.cpp)
target_include_directories(test_cli PRIVATE tests src)
add_test(NAME cli COMMAND test_cli)

add_library(mxbm_config STATIC src/config/config.cpp)
target_include_directories(mxbm_config PUBLIC src)
target_link_libraries(mxbm_config PUBLIC mxbm_json)

add_executable(test_config tests/test_config.cpp)
target_include_directories(test_config PRIVATE tests src)
target_link_libraries(test_config PRIVATE mxbm_config)
target_compile_definitions(test_config PRIVATE MXBM_FIXTURES="${CMAKE_SOURCE_DIR}/tests/fixtures")
add_test(NAME config COMMAND test_config)

add_library(mxbm_ui STATIC src/ui/console.cpp src/ui/format.cpp src/ui/ticker.cpp)
target_include_directories(mxbm_ui PUBLIC src)
target_link_libraries(mxbm_ui PUBLIC mxbm_miner mxbm_version)

add_executable(test_format tests/test_format.cpp)
target_include_directories(test_format PRIVATE tests src)
target_link_libraries(test_format PRIVATE mxbm_ui)
add_test(NAME format COMMAND test_format)

add_executable(test_benchmark tests/test_benchmark.cpp)
target_include_directories(test_benchmark PRIVATE tests src)
target_link_libraries(test_benchmark PRIVATE mxbm_miner mxbm_ui mxbm_stratum mxbm_pow)
add_test(NAME benchmark COMMAND test_benchmark)

add_executable(test_console tests/test_console.cpp)
target_include_directories(test_console PRIVATE tests src)
target_link_libraries(test_console PRIVATE mxbm_ui)
add_test(NAME console COMMAND test_console)

add_library(mxbm_api STATIC src/api/http_summary.cpp)
target_include_directories(mxbm_api PUBLIC src)
target_link_libraries(mxbm_api PUBLIC mxbm_miner mxbm_json mxbm_version)

add_executable(test_api tests/test_api.cpp)
target_include_directories(test_api PRIVATE tests src)
target_link_libraries(test_api PRIVATE mxbm_api Threads::Threads)
add_test(NAME api COMMAND test_api)

# NVML telemetry: dlopen'd at runtime, so no build-time NVIDIA dependency. Compiled
# unconditionally -- it reports nothing on a machine without the driver.
add_library(mxbm_nvml STATIC src/gpu/nvml.cpp)
target_include_directories(mxbm_nvml PUBLIC src)
target_link_libraries(mxbm_nvml PUBLIC ${CMAKE_DL_LIBS})

add_executable(mxbm src/main.cpp src/cli/options.cpp)
target_link_libraries(mxbm PRIVATE mxbm_nvml)
target_include_directories(mxbm PRIVATE src)
target_link_libraries(mxbm PRIVATE mxbm_miner)
target_link_libraries(mxbm PRIVATE mxbm_ui)
target_link_libraries(mxbm PRIVATE mxbm_version)
target_link_libraries(mxbm PRIVATE mxbm_config)
target_link_libraries(mxbm PRIVATE mxbm_api)
if(MXBM_HAVE_BEAM_ORACLE)
  target_compile_definitions(mxbm PRIVATE MXBM_HAVE_BEAM_ORACLE)
endif()

set(MXBM_VERSION_HEADER ${CMAKE_BINARY_DIR}/generated/mxbm_version.h)
add_custom_target(mxbm_version_gen
    COMMAND ${CMAKE_COMMAND}
        -DSRC_DIR=${CMAKE_SOURCE_DIR} -DOUT_FILE=${MXBM_VERSION_HEADER}
        -DVER_MAJ=${PROJECT_VERSION_MAJOR} -DVER_MIN=${PROJECT_VERSION_MINOR}
        -P ${CMAKE_SOURCE_DIR}/cmake/GenerateVersion.cmake
    BYPRODUCTS ${MXBM_VERSION_HEADER}
    COMMENT "Stamping mxbm_version.h")
add_library(mxbm_version STATIC src/version.cpp)
target_include_directories(mxbm_version PUBLIC src PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_dependencies(mxbm_version mxbm_version_gen)

add_executable(test_version tests/test_version.cpp)
target_include_directories(test_version PRIVATE tests src)
target_link_libraries(test_version PRIVATE mxbm_version)
add_test(NAME version COMMAND test_version)

# ---- CUDA GPU solver (optional component) ----
# Optional in exactly the same way OpenCL is: if nvcc is absent the build proceeds
# without it and the OpenCL path remains the default. Kept separate so an incomplete or
# broken CUDA toolchain cannot break a build that only wants OpenCL.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  message(STATUS "CUDA found: CUDA solver enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
  add_library(mxbm_cuda STATIC src/gpu/cuda_solver.cu)
  target_include_directories(mxbm_cuda PUBLIC src PRIVATE kernels/cuda)
  target_link_libraries(mxbm_cuda PUBLIC mxbm_beamhash)
  set_target_properties(mxbm_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  # 186 = "pointless comparison of unsigned integer with zero", from template
  # instantiations where a leaf count is 0 and the loop compiles away.
  target_compile_options(mxbm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=186>)
  target_compile_features(mxbm_cuda PRIVATE cuda_std_17)
  target_link_libraries(mxbm PRIVATE mxbm_cuda)
  target_compile_definitions(mxbm PRIVATE MXBM_HAVE_CUDA)

  add_executable(test_cuda_solver tests/test_cuda_solver.cu)
  target_include_directories(test_cuda_solver PRIVATE tests src)
  target_link_libraries(test_cuda_solver PRIVATE mxbm_cuda)
  target_compile_features(test_cuda_solver PRIVATE cuda_std_17)
  add_test(NAME cuda_solver COMMAND test_cuda_solver)
else()
  message(STATUS "CUDA not found: CUDA solver disabled (OpenCL path unaffected)")
endif()

# ---- M3: OpenCL GPU solver (optional component) ----
find_package(OpenCL)
if(OpenCL_FOUND)
  message(STATUS "OpenCL found: GPU targets enabled")
  set(MXBM_HAVE_OPENCL ON)

  add_library(mxbm_gpu STATIC src/gpu/cl_runtime.cpp)
  target_include_directories(mxbm_gpu PUBLIC src)
  target_include_directories(mxbm_gpu PRIVATE ${CMAKE_BINARY_DIR})
  target_link_libraries(mxbm_gpu PUBLIC OpenCL::OpenCL)
  target_link_libraries(mxbm_gpu PUBLIC mxbm_beamhash)   # gpu_solver.cpp: is_valid_solution/pack_indices/compute_prepow
  target_sources(mxbm_gpu PRIVATE src/gpu/budget.cpp)
  target_sources(mxbm_gpu PRIVATE src/gpu/round_pipeline.cpp)
  target_sources(mxbm_gpu PRIVATE src/gpu/gpu_solver.cpp)

  # Wire the GPU solver into the mxbm binary itself (mxbm target already
  # exists -- defined above -- so this just adds mxbm_gpu's PUBLIC
  # include dirs / OpenCL::OpenCL / mxbm_beamhash link requirements to it).
  target_link_libraries(mxbm PRIVATE mxbm_gpu)
  target_compile_definitions(mxbm PRIVATE MXBM_HAVE_OPENCL)

  add_executable(test_cl_smoke tests/test_cl_smoke.cpp)
  target_include_directories(test_cl_smoke PRIVATE tests src)
  target_link_libraries(test_cl_smoke PRIVATE mxbm_gpu)
  add_test(NAME cl_smoke COMMAND test_cl_smoke)

  add_executable(test_budget tests/test_budget.cpp)
  target_include_directories(test_budget PRIVATE tests src)
  target_link_libraries(test_budget PRIVATE mxbm_gpu)
  add_test(NAME budget COMMAND test_budget)

  add_executable(test_round_reconstruct tests/test_round_reconstruct.cpp)
  target_include_directories(test_round_reconstruct PRIVATE tests src src/beamhash)
  target_link_libraries(test_round_reconstruct PRIVATE mxbm_gpu)
  add_test(NAME round_reconstruct COMMAND test_round_reconstruct)

  set(MXBM_KERNELS_HEADER ${CMAKE_BINARY_DIR}/generated/mxbm_kernels.h)
  set(MXBM_KERNEL_FILES
      ${CMAKE_SOURCE_DIR}/kernels/opencl/bh3.cl
      ${CMAKE_SOURCE_DIR}/kernels/opencl/seed.cl
      ${CMAKE_SOURCE_DIR}/kernels/opencl/round.cl
      ${CMAKE_SOURCE_DIR}/kernels/opencl/sort.cl
      ${CMAKE_SOURCE_DIR}/kernels/opencl/lds.cl)
  add_custom_command(
    OUTPUT ${MXBM_KERNELS_HEADER}
    COMMAND ${CMAKE_COMMAND} "-DKERNEL_FILES=${MXBM_KERNEL_FILES}"
            -DOUT_FILE=${MXBM_KERNELS_HEADER}
            -P ${CMAKE_SOURCE_DIR}/cmake/EmbedKernels.cmake
    DEPENDS ${MXBM_KERNEL_FILES} ${CMAKE_SOURCE_DIR}/cmake/EmbedKernels.cmake
    COMMENT "Embedding OpenCL kernels -> mxbm_kernels.h"
    VERBATIM)
  add_custom_target(mxbm_kernels_gen DEPENDS ${MXBM_KERNELS_HEADER})
  add_dependencies(mxbm_gpu mxbm_kernels_gen)   # round_pipeline.cpp includes the generated header

  add_executable(test_cl_primitives tests/test_cl_primitives.cpp)
  add_dependencies(test_cl_primitives mxbm_kernels_gen)
  target_include_directories(test_cl_primitives PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_cl_primitives PRIVATE mxbm_gpu)
  add_test(NAME cl_primitives COMMAND test_cl_primitives)

  add_executable(test_gpu_seed tests/test_gpu_seed.cpp)
  add_dependencies(test_gpu_seed mxbm_kernels_gen)
  target_include_directories(test_gpu_seed PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_gpu_seed PRIVATE mxbm_gpu)
  add_test(NAME gpu_seed COMMAND test_gpu_seed)

  # Phase-D3 P1: sort-based collision-finder building blocks (kernels/opencl/
  # sort.cl) verified in isolation vs CPU std::sort / group-by-key oracles.
  add_executable(test_gpu_sort tests/test_gpu_sort.cpp)
  add_dependencies(test_gpu_sort mxbm_kernels_gen)
  target_include_directories(test_gpu_sort PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_gpu_sort PRIVATE mxbm_gpu)
  add_test(NAME gpu_sort COMMAND test_gpu_sort)

  # P2a: LDS-local bucketed collision finder (kernels/opencl/lds.cl) validated in
  # isolation vs a CPU group-by-key oracle + an empirical fork sweep.
  add_executable(test_lds_collide tests/test_lds_collide.cpp)
  add_dependencies(test_lds_collide mxbm_kernels_gen)
  target_include_directories(test_lds_collide PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_lds_collide PRIVATE mxbm_gpu)
  add_test(NAME lds_collide COMMAND test_lds_collide)

  add_executable(test_round_mix tests/test_round_mix.cpp)
  add_dependencies(test_round_mix mxbm_kernels_gen)
  target_include_directories(test_round_mix PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_round_mix PRIVATE mxbm_gpu)
  add_test(NAME round_mix COMMAND test_round_mix)
  # White-box tests hand-drive the flat-work[] sort-path kernels via alloc_pipeline;
  # pin them off the device-mem-auto row-bucket default (which omits work[]/leaves[]).
  set_tests_properties(round_mix PROPERTIES ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  add_executable(test_round_scatter tests/test_round_scatter.cpp)
  add_dependencies(test_round_scatter mxbm_kernels_gen)
  target_include_directories(test_round_scatter PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_round_scatter PRIVATE mxbm_gpu)
  add_test(NAME round_scatter COMMAND test_round_scatter)
  set_tests_properties(round_scatter PROPERTIES ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  add_executable(test_round_match tests/test_round_match.cpp)
  add_dependencies(test_round_match mxbm_kernels_gen)
  target_include_directories(test_round_match PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_round_match PRIVATE mxbm_gpu)
  add_test(NAME round_match COMMAND test_round_match)
  # test_round_match validates the LEGACY scatter+match kernels directly and its
  # run_single_round() smoke sections upload pre-mixed work (no fused sort pair),
  # so pin those to the legacy collision path (the sort path is now the default).
  set_tests_properties(round_match PROPERTIES ENVIRONMENT "MXBM_LEGACY_MATCH=1")

  add_executable(test_round_pipeline tests/test_round_pipeline.cpp)
  add_dependencies(test_round_pipeline mxbm_kernels_gen)
  target_include_directories(test_round_pipeline PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_round_pipeline PRIVATE mxbm_gpu)
  add_test(NAME round_pipeline COMMAND test_round_pipeline)
  set_tests_properties(round_pipeline PROPERTIES ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  add_executable(test_gpu_rounds tests/test_gpu_rounds.cpp)
  add_dependencies(test_gpu_rounds mxbm_kernels_gen)
  target_include_directories(test_gpu_rounds PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_gpu_rounds PRIVATE mxbm_gpu)
  add_test(NAME gpu_rounds COMMAND test_gpu_rounds)
  # Phase-B exit milestone: a real full 2^25 GPU Wagner search (5 rounds,
  # ~2^30 key-compares/round) -- compute-heavy but observed ~3s on an M3 Max;
  # generous timeout so a slow-but-correct run on lesser hardware isn't killed.
  # Pin the bare gate to the sort path (deterministic across hardware); the fused
  # row-bucket default is auto-selected by device memory and covered by the
  # gpu_*_rowbucket gates below.
  set_tests_properties(gpu_rounds PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  add_executable(test_gpu_recover tests/test_gpu_recover.cpp)
  add_dependencies(test_gpu_recover mxbm_kernels_gen)
  target_include_directories(test_gpu_recover PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_gpu_recover PRIVATE mxbm_gpu)
  add_test(NAME gpu_recover COMMAND test_gpu_recover)
  # Phase-C Task 1: runs the same full 2^25 GPU Wagner search as gpu_rounds
  # (~2-3s on an M3 Max) plus a cheap on-device recovery pass; same generous
  # timeout as gpu_rounds for the same reason (slow-but-correct hardware).
  set_tests_properties(gpu_recover PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  add_executable(test_gpu_solver tests/test_gpu_solver.cpp)
  add_dependencies(test_gpu_solver mxbm_kernels_gen)
  target_include_directories(test_gpu_solver PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(test_gpu_solver PRIVATE mxbm_gpu)
  add_test(NAME gpu_solver COMMAND test_gpu_solver)
  # Phase-C Task 2 -- THE MILESTONE: GpuSolver::solve() runs the same full
  # 2^25 GPU Wagner search as gpu_rounds/gpu_recover (~2-3s on an M3 Max)
  # plus recovery plus the CPU verify gate; same generous timeout for the
  # same reason (slow-but-correct hardware).
  set_tests_properties(gpu_solver PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_NO_ROWBUCKET=1")

  # Phase-D3: the sort-based collision finder (key_extract -> radix sort ->
  # run-scan emit) is now the DEFAULT, so the plain gpu_rounds/gpu_solver gates
  # above already exercise it. These extra gates re-run the SAME golden binaries
  # through the LEGACY atomic-bucket scatter+match (MXBM_LEGACY_MATCH=1) so both
  # collision paths stay byte-identical-to-goldens under CI. Reuses the existing
  # executables -- just an env-var-flipped invocation held to the suite.
  add_test(NAME gpu_rounds_legacy COMMAND test_gpu_rounds)
  set_tests_properties(gpu_rounds_legacy PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_LEGACY_MATCH=1")
  add_test(NAME gpu_solver_legacy COMMAND test_gpu_solver)
  set_tests_properties(gpu_solver_legacy PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_LEGACY_MATCH=1")
  # P2c: the SAME goldens through the LDS-local match (round_scatter_lds ->
  # round_collide_lds). Byte-identical is the equivalence gate for the rewrite.
  add_test(NAME gpu_rounds_lds COMMAND test_gpu_rounds)
  set_tests_properties(gpu_rounds_lds PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_LDS_MATCH=1")
  add_test(NAME gpu_solver_lds COMMAND test_gpu_solver)
  set_tests_properties(gpu_solver_lds PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_LDS_MATCH=1")
  # P2 STEP C: the SAME goldens through the FUSED row-bucket pipeline (round1_mix_
  # scatter_fat -> round_fused_lds x4 -> round5_fused_lds). Byte-identical goldens
  # are the equivalence gate; the win is ~215 -> ~148 ms/solve.
  add_test(NAME gpu_rounds_rowbucket COMMAND test_gpu_rounds)
  set_tests_properties(gpu_rounds_rowbucket PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_ROWBUCKET=1")
  add_test(NAME gpu_solver_rowbucket COMMAND test_gpu_solver)
  set_tests_properties(gpu_solver_rowbucket PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_ROWBUCKET=1")
  add_test(NAME gpu_recover_rowbucket COMMAND test_gpu_recover)
  set_tests_properties(gpu_recover_rowbucket PROPERTIES TIMEOUT 1800 ENVIRONMENT "MXBM_ROWBUCKET=1")

  # Phase-D benchmark harness (manual-run, NOT registered as a ctest gate):
  # repeats the full 2^25 solve N times, enforces the D0 correctness gate,
  # and reports median total + per-phase ms for comparing match-kernel variants.
  add_executable(bench_rounds tests/bench_rounds.cpp)
  add_dependencies(bench_rounds mxbm_kernels_gen)
  target_include_directories(bench_rounds PRIVATE tests src src/beamhash ${CMAKE_BINARY_DIR})
  target_link_libraries(bench_rounds PRIVATE mxbm_gpu)
else()
  message(STATUS "OpenCL not found: GPU targets skipped (core suite unaffected)")
endif()
