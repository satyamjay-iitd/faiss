/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace faiss {

/// Deflate-compress a byte buffer (zlib's one-shot `compress2` API, level 9
/// -- these buffers are small, one per segment, so streaming isn't needed).
/// Stands in for JPEG's own entropy-coding step (Huffman/arithmetic coding
/// on a zigzag-ordered, run-length-encoded stream): quantization already
/// concentrates most of the byte stream into runs of zero, and this is
/// where that gets turned into an actual space saving.
std::vector<uint8_t> zlib_compress(const uint8_t* data, size_t size);

/// Inverse of zlib_compress. `expected_size` (the original, uncompressed
/// size) must be known by the caller -- it's not embedded in the stream --
/// and the output buffer must already be sized to hold it.
void zlib_decompress(
        const uint8_t* compressed,
        size_t compressed_size,
        uint8_t* out,
        size_t expected_size);

} // namespace faiss
