#pragma once

// The long-prompt layer skip of DeepSeek-V4.1 (T0829), after Strata's "YOCO" skip. From the first
// block with compression ratio 1 (`skip_layer`, 20 in the Flash release) on, every block is ratio 1
// and reads the compressed latents and indexer keys of that block alone (it is the last kv source);
// the blocks after it own a sliding window and nothing else a prompt row leaves behind. A row's
// output at block l therefore depends on the block's own window (the `window - 1` rows before it) and
// the kv source's compressed cache, and the rows a prompt must run through every block are only the
// ones its last window, the decode's rings, and the next checkpoint depend on: the last
// `keep = (layers - skip_layer) * (window - 1) + window` rows before the point the pass ends at.
// Earlier rows run the blocks before `skip_layer` and, at `skip_layer`, only the compressor and
// the indexer's keys (kv_only): no attention, no FFN, no expert is delivered for them in the
// blocks from there on.
//
// Exactness: row r of block l reads the latents of rows r - window + 1 .. r of block l; the rows
// before the cut have none at the blocks past `skip_layer`, so a block's first `window - 1` rows
// past the cut are computed over a window of unwritten latents, and that damage widens by
// `window - 1` rows a block: block skip_layer + k is damaged in the rows below
// cut + (k + 1) * (window - 1). The rings and the logits read the last `window` rows of each block's
// input, which lie past the damage of the block before it when the pass ends at least
// (layers - skip_layer - 1) * (window - 1) + window rows past the cut; keep is one window more, as
// Strata's. The state at any point before the pass's end is therefore not exact, and a checkpoint is
// only taken at the end of a pass (a frontier).
//
// Everything here is plain arithmetic over the config, shared by the executor and the engine and
// checked on the CPU (tests/models/deepseek_v41/test_prompt_skip.cpp).

#include "models/deepseek_v41/config.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::models::deepseek_v41 {

// A skip of fewer rows costs more than it saves (the kv_only pass and the cut's boundary).
inline constexpr std::uint32_t kPromptSkipMinRows = 1024;
// Rows a pass keeps through every block at least when it keeps any: fewer rows make narrow expert
// calls in each of the blocks past the cut, which cost a block's copies for a few rows.
inline constexpr std::uint32_t kPromptSkipMinKept = 512;

// The block the skip starts at, or -1 when the config does not have the structure: the first block
// with ratio 1 is a kv source after at least one other block, and every block after it has ratio 1
// and no compressed cache or keys of its own.
[[nodiscard]] inline int prompt_skip_layer(const TextConfig& c) noexcept {
    const std::size_t layers = c.num_hidden_layers;
    if (layers < 2 || c.compress_ratios.size() != layers || c.kv_source.size() != layers ||
        c.index_source.size() != layers || c.sliding_window == 0) {
        return -1;
    }
    std::size_t first = layers;
    for (std::size_t l = 0; l < layers; ++l) {
        if (c.compress_ratios[l] == 1) {
            first = l;
            break;
        }
    }
    if (first == 0 || first >= layers || !c.kv_source[first]) { return -1; }
    for (std::size_t l = first + 1; l < layers; ++l) {
        if (c.compress_ratios[l] != 1 || c.kv_source[l]) { return -1; }
    }
    return int(first);
}

// Rows before a pass's end that go through every block: (layers - skip_layer) windows of
// `window - 1` rows of damage and one window. 0 when there is no skip.
[[nodiscard]] inline std::uint32_t prompt_skip_keep(int skip_layer, std::uint32_t layers,
                                                    std::uint32_t window) noexcept {
    if (skip_layer <= 0 || std::uint32_t(skip_layer) >= layers || window == 0) { return 0; }
    return (layers - std::uint32_t(skip_layer)) * (window - 1) + window;
}

// The rows at the start of a pass of `step` prompt tokens that skip the blocks past `skip_layer`,
// when the pass belongs to a prompt run that ends `to_frontier` tokens after the pass starts (the
// next point a state is kept at: a checkpoint or the prompt's end; at least `step`). 0: no skip.
// The result is `step` for a pass wholly before the kept rows, else the rows before them, rounded
// down so that at least kPromptSkipMinKept rows are kept.
[[nodiscard]] inline std::uint32_t plan_prompt_skip(std::uint32_t step, std::uint32_t to_frontier,
                                                    std::uint32_t keep) noexcept {
    if (keep == 0 || step == 0 || to_frontier < step || to_frontier <= keep) { return 0; }
    std::uint32_t skip = std::min(step, to_frontier - keep);
    if (skip < step && step - skip < kPromptSkipMinKept) {
        skip = step > kPromptSkipMinKept ? step - kPromptSkipMinKept : 0;
    }
    return skip >= kPromptSkipMinRows ? skip : 0;
}

// The prompt chunks the engine tries, widest first, for a requested --prefill-chunk (0: none):
// 2048 then 1024 and 512 as before; a request names a first try (a multiple of 128, at most the
// executor's 4096, past which it is clamped) and those of 2048, 1024 and 512 below it follow.
[[nodiscard]] inline std::vector<std::uint32_t> prompt_chunk_candidates(std::uint32_t requested) {
    constexpr std::uint32_t kLargest = 4096;
    std::vector<std::uint32_t> out;
    if (requested == 0) {
        out = {2048, 1024, 512};
        return out;
    }
    const std::uint32_t first = std::min(requested / 128 * 128, kLargest);
    out.push_back(first == 0 ? 128 : first);
    for (const std::uint32_t chunk : {2048u, 1024u, 512u}) {
        if (chunk < out.front()) { out.push_back(chunk); }
    }
    return out;
}

// A run of rows of one block in a span pass: rows [row, row + rows) as one chunk; `kv_only` rows
// run the block's compressor and indexer keys alone.
struct SkipPiece {
    int row      = 0;
    int rows     = 0;
    bool kv_only = false;
};

// The chunks of block `layer` over a span of `n` rows split in chunks of `chunk`, the first `skip`
// of which skip the blocks past `skip_layer`: every block before it runs the span in chunks from
// row 0 as without a skip; the block itself runs rows [0, skip) in chunks from row 0 as kv_only
// pieces, and the rows from skip in chunks from there; the blocks after it run the rows from skip
// alone. `skip_layer < 0` or `skip == 0`: the plain split.
inline void plan_skip_pieces(int layer, int skip_layer, int n, int chunk, int skip,
                             std::vector<SkipPiece>& out) {
    out.clear();
    const auto split = [&](int begin, int end, bool kv_only) {
        for (int row = begin; row < end; row += chunk) {
            out.push_back({row, std::min(chunk, end - row), kv_only});
        }
    };
    if (skip_layer < 0 || skip <= 0 || layer < skip_layer) {
        split(0, n, false);
        return;
    }
    if (layer == skip_layer) { split(0, skip, true); }
    split(skip, n, false);
}

} // namespace ninfer::models::deepseek_v41
