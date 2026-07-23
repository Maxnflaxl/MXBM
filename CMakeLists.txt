cmake_minimum_required(VERSION 3.16)
project(MXBM VERSION 0.2 LANGUAGES C CXX)
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

add_library(mxbm_miner STATIC src/miner/engine.cpp)
target_include_directories(mxbm_miner PUBLIC src)
target_link_libraries(mxbm_miner PUBLIC mxbm_stratum mxbm_pow)
target_sources(mxbm_miner PRIVATE src/miner/stats.cpp)

if(MXBM_HAVE_BEAM_ORACLE)
  target_sources(mxbm_miner PRIVATE src/miner/solver_ref.cpp)
  target_link_libraries(mxbm_miner PUBLIC beam_oracle)
endif()

add_executable(test_engine tests/test_engine.cpp)
target_include_directories(test_engine PRIVATE tests src)
target_link_libraries(test_engine PRIVATE mxbm_miner)
add_test(NAME engine COMMAND test_engine)

add_executable(test_stats tests/test_stats.cpp)
target_include_directories(test_stats PRIVATE tests src)
target_link_libraries(test_stats PRIVATE mxbm_miner)
add_test(NAME stats COMMAND test_stats)

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

add_library(mxbm_api STATIC src/api/http_summary.cpp)
target_include_directories(mxbm_api PUBLIC src)
target_link_libraries(mxbm_api PUBLIC mxbm_miner mxbm_json mxbm_version)

add_executable(test_api tests/test_api.cpp)
target_include_directories(test_api PRIVATE tests src)
target_link_libraries(test_api PRIVATE mxbm_api Threads::Threads)
add_test(NAME api COMMAND test_api)

add_executable(mxbm src/main.cpp src/cli/options.cpp)
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
