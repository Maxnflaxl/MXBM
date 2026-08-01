# Embed an arbitrary binary file as a C array, so mxbm stays a single
# self-contained binary with no runtime file lookup.
# Invoked as: cmake -DIN_FILE=... -DOUT_FILE=... -P EmbedBinary.cmake
file(READ ${IN_FILE} _hex HEX)
string(LENGTH "${_hex}" _hexlen)
math(EXPR _len "${_hexlen} / 2")
# One "0xNN," per byte. Line-wrapped every 32 bytes: a single multi-megabyte
# line makes some compilers slow and every diff tool useless.
string(REGEX REPLACE "(................................................................)"
       "\\1\n" _wrapped "${_hex}")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _arr "${_wrapped}")
file(WRITE ${OUT_FILE}
  "// Generated from ${IN_FILE} -- do not edit.\n"
  "#pragma once\n"
  "extern \"C\" {\n"
  "inline const unsigned char kMxbmMetallib[] = {\n${_arr}\n};\n"
  "inline const unsigned long kMxbmMetallibSize = ${_len};\n"
  "}\n")
