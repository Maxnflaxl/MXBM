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
function(mxbm_build_metallib OUT_LIB)
  set(_airs "")
  foreach(_src ${ARGN})
    get_filename_component(_name ${_src} NAME_WE)
    set(_air ${CMAKE_BINARY_DIR}/generated/${_name}.air)
    add_custom_command(OUTPUT ${_air}
      COMMAND xcrun metal -c ${_src} -o ${_air}
              -I ${CMAKE_SOURCE_DIR}/src
              -I ${CMAKE_SOURCE_DIR}/src/beamhash
              -O3
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
