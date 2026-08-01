# Vendored Blake2b — the single Blake2b in the whole build.
add_library(mxbm_blake2b STATIC third_party/blake2b/blake2b-ref.c)
target_include_directories(mxbm_blake2b PUBLIC third_party/blake2b)

# Optional Beam reference island for the differential test. Probed rather than
# defaulted to one machine's layout, so the oracle is not silently OFF elsewhere;
# -DBEAM_SOURCE_DIR=... wins, being probed first.
set(BEAM_SOURCE_DIR "" CACHE PATH "Beam source root (auto-probed when empty)")
set(_beam_candidates
    "${BEAM_SOURCE_DIR}"
    "$ENV{HOME}/Developer/Github/Beam/beam")
set(_bh3 "")
foreach(_cand IN LISTS _beam_candidates)
  if(_cand AND EXISTS "${_cand}/3rdparty/crypto/beamHashIII_impl.cpp")
    set(_beam_root "${_cand}")
    set(_bh3 "${_cand}/3rdparty/crypto/beamHashIII_impl.cpp")
    break()
  endif()
endforeach()
if(_bh3)
  add_library(beam_oracle STATIC "${_bh3}")
  # Compile ONLY beamHashIII_impl.cpp; blake symbols come from mxbm_blake2b.
  target_include_directories(beam_oracle PUBLIC "${_beam_root}/3rdparty/crypto")
  target_link_libraries(beam_oracle PUBLIC mxbm_blake2b)
  # beamHashIII.h only *declares* OptimisedSolve under ENABLE_MINING, but
  # beamHashIII_impl.cpp *defines* it unconditionally -- an out-of-line
  # definition with no matching declaration unless the macro is set. Beam's
  # own build (pow/CMakeLists.txt) always compiles this file with
  # ENABLE_MINING; PUBLIC so consumers (e.g. test_oracle_spike) see the same
  # class layout -- BeamHash_III is polymorphic (inherits PoWScheme), so a
  # mismatch here would be a real ODR / vtable-layout hazard, not just a
  # missing symbol.
  target_compile_definitions(beam_oracle PUBLIC ENABLE_MINING)
  set(MXBM_HAVE_BEAM_ORACLE ON CACHE INTERNAL "")
  message(STATUS "Beam differential oracle: ENABLED (${_beam_root})")
else()
  set(MXBM_HAVE_BEAM_ORACLE OFF CACHE INTERNAL "")
  message(STATUS "Beam differential oracle: DISABLED (set -DBEAM_SOURCE_DIR=...)")
endif()
