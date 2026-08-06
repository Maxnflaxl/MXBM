# Build-time version stamp: v<MAJ>.<MIN>.<git commit count> [<short hash>]
# Invoked with -DSRC_DIR= -DOUT_FILE= -DVER_MAJ= -DVER_MIN= -P this-file.
execute_process(COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY "${SRC_DIR}" OUTPUT_VARIABLE GIT_COUNT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE RC1)
execute_process(COMMAND git rev-parse --short=7 HEAD
    WORKING_DIRECTORY "${SRC_DIR}" OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE RC2)
if(NOT RC1 EQUAL 0 OR NOT RC2 EQUAL 0 OR GIT_COUNT STREQUAL "" OR GIT_HASH STREQUAL "")
    set(GIT_COUNT 0)
    set(GIT_HASH "nogit")
endif()
set(CONTENT "#pragma once
#define MXBM_VERSION_STRING \"v${VER_MAJ}.${VER_MIN}.${GIT_COUNT} [${GIT_HASH}]\"
")
set(OLD "")
if(EXISTS "${OUT_FILE}")
    file(READ "${OUT_FILE}" OLD)
endif()
if(NOT OLD STREQUAL CONTENT)          # rewrite only on change: no rebuild churn
    file(WRITE "${OUT_FILE}" "${CONTENT}")
endif()
