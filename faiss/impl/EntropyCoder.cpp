/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/EntropyCoder.h>

#include <zlib.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

std::vector<uint8_t> zlib_compress(const uint8_t* data, size_t size) {
    if (size == 0) {
        return {};
    }
    uLongf bound = compressBound(static_cast<uLong>(size));
    std::vector<uint8_t> out(bound);
    uLongf out_size = bound;
    int rc = compress2(
            out.data(),
            &out_size,
            data,
            static_cast<uLong>(size),
            /*level=*/9);
    FAISS_THROW_IF_NOT_MSG(rc == Z_OK, "zlib_compress: compress2 failed");
    out.resize(out_size);
    return out;
}

void zlib_decompress(
        const uint8_t* compressed,
        size_t compressed_size,
        uint8_t* out,
        size_t expected_size) {
    if (expected_size == 0) {
        return;
    }
    uLongf out_size = static_cast<uLongf>(expected_size);
    int rc = uncompress(
            out, &out_size, compressed, static_cast<uLong>(compressed_size));
    FAISS_THROW_IF_NOT_MSG(rc == Z_OK, "zlib_decompress: uncompress failed");
    FAISS_THROW_IF_NOT_MSG(
            out_size == expected_size,
            "zlib_decompress: decompressed size did not match expected_size");
}

} // namespace faiss
