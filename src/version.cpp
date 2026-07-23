// version.cpp — the only TU that rebuilds when the stamp changes
#include "version.h"
#include "mxbm_version.h"   // generated into ${CMAKE_BINARY_DIR}/generated
namespace mxbm {
const char* version() { return MXBM_VERSION_STRING; }
}
