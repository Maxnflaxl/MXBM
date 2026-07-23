#pragma once
/*
 * Public-domain SHA-256 (FIPS 180-4), single-shot API.
 * See NOTICE in this directory.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Computes the 32-byte SHA-256 digest of data[0..len) into out[0..32). */
void sha256(const uint8_t* data, size_t len, uint8_t out[32]);

#ifdef __cplusplus
}
#endif
