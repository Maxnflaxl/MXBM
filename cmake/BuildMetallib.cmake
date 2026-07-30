# Compile kernels/metal/*.metal into a single .metallib.
#
# Requires the Metal toolchain, which ships SEPARATELY from Xcode:
#   xcodebuild -downloadComponent MetalToolchain
# Absent it, the caller disables the Metal backend entirely and every other
# target still builds -- exactly how the CUDA and OpenCL blocks behave.
#
# Kernels are compiled AHEAD of time rather than from source at startup so a
# syntax error fails the build instead of the miner, and so the runtime cost of
# newLibraryWithSource: is not paid on every launch.
# Sweepable tuning constants. The Metal kernels inherit kWG / kFCap / kR1FCap from
# Ada, where they were tuned against an occupancy model that does not transfer, so
# they are build-time knobs here exactly as MXBM_WG / MXBM_FCAP are on the CUDA side.
set(MXBM_METAL_WG      "" CACHE STRING "Metal threadgroup size (default 256)")
set(MXBM_METAL_FCAP    "" CACHE STRING "Metal per-group stage cap (default 320)")
set(MXBM_METAL_R1FCAP  "" CACHE STRING "Metal round-1 stage cap (default 288)")
# Attribution build. RESULTS ARE INTENTIONALLY WRONG when non-zero; the goldens will
# fail, which is the point -- it isolates what a stage costs, it does not mine.
set(MXBM_METAL_ABL_DERIVE "" CACHE STRING "Ablate a re-derivation (2 = round 2's rebuild)")

function(mxbm_build_metallib OUT_LIB)
  set(MXBM_METAL_DEFS "")
  foreach(_k WG FCAP R1FCAP ABL_DERIVE)
    if(NOT MXBM_METAL_${_k} STREQUAL "")
      list(APPEND MXBM_METAL_DEFS -DMXBM_METAL_${_k}=${MXBM_METAL_${_k}})
    endif()
  endforeach()
  if(MXBM_METAL_DEFS)
    message(STATUS "Metal tuning: ${MXBM_METAL_DEFS}")
  endif()
  set(_airs "")
  foreach(_src ${ARGN})
    get_filename_component(_name ${_src} NAME_WE)
    set(_air ${CMAKE_BINARY_DIR}/generated/${_name}.air)
    add_custom_command(OUTPUT ${_air}
      COMMAND xcrun metal -c ${_src} -o ${_air}
              -I ${CMAKE_SOURCE_DIR}/src
              -I ${CMAKE_SOURCE_DIR}/src/beamhash
              -O3 ${MXBM_METAL_DEFS}
      DEPENDS ${_src}
              ${CMAKE_SOURCE_DIR}/src/beamhash/bh3_primitives.h
              ${CMAKE_SOURCE_DIR}/src/beamhash/bh3_types.h
      COMMENT "metal -c ${_name}.metal"
      VERBATIM)
    list(APPEND _airs ${_air})
  endforeach()
  add_custom_command(OUTPUT ${OUT_LIB}
    COMMAND xcrun metallib ${_airs} -o ${OUT_LIB}
    DEPENDS ${_airs}
    COMMENT "metallib -> mxbm.metallib"
    VERBATIM)
endfunction()
