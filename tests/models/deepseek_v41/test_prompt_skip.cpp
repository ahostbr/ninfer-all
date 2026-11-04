// CPU checks of the long-prompt layer skip's arithmetic (src/models/deepseek_v41/prompt_skip.h):
// which block it starts at, how many rows a run keeps, how a step is cut, how a block's chunks are
// split around the cut, and a dependency model of the exactness argument. No device is used.

#include "models/deepseek_v41/prompt_skip.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer::models::deepseek_v41;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The Flash release's block structure: blocks 0-1 window only, 2-19 ratio 2, 20-39 ratio 1; kv
// sources 2, 8, 14 and 20; index sources 2, 8, 14, 20 and every fourth after.
TextConfig release() {
    TextConfig c;
    c.num_hidden_layers = 40;
    c.sliding_window    = 128;
    c.compress_ratios.assign(40, 0);
    c.kv_source.assign(40, false);
    c.index_source.assign(40, false);
    for (int l = 2; l < 20; ++l) { c.compress_ratios[std::size_t(l)] = 2; }
    for (int l = 20; l < 40; ++l) { c.compress_ratios[std::size_t(l)] = 1; }
    for (const int l : {2, 8, 14, 20}) { c.kv_source[std::size_t(l)] = true; }
    for (const int l : {2, 8, 14, 20, 24, 28, 32, 36}) { c.index_source[std::size_t(l)] = true; }
    c.candidate_source = 20;
    return c;
}

int test_layer_and_keep() {
    int failures = 0;
    const TextConfig c = release();
    failures += check(prompt_skip_layer(c) == 20, "the release's skip block is not 20");
    failures += check(prompt_skip_keep(20, 40, 128) == 2668,
                      "the release's keep is not (40 - 20) * 127 + 128 = 2668");
    failures += check(prompt_skip_keep(-1, 40, 128) == 0 && prompt_skip_keep(0, 40, 128) == 0 &&
                          prompt_skip_keep(40, 40, 128) == 0 && prompt_skip_keep(20, 40, 0) == 0,
                      "a keep for no skip block");
    {
        TextConfig b = c; // ratio 1 from block 0: nothing before the skip block
        for (auto& r : b.compress_ratios) { r = 1; }
        failures += check(prompt_skip_layer(b) == -1, "a skip block at block 0");
    }
    {
        TextConfig b = c; // a kv source past the skip block owns a cache of its own
        b.kv_source[30] = true;
        failures += check(prompt_skip_layer(b) == -1, "a kv source past the skip block");
    }
    {
        TextConfig b = c; // a ratio-2 block past it
        b.compress_ratios[33] = 2;
        failures += check(prompt_skip_layer(b) == -1, "a ratio-2 block past the skip block");
    }
    {
        TextConfig b = c; // a window-only block past it
        b.compress_ratios[39] = 0;
        failures += check(prompt_skip_layer(b) == -1, "a window-only block past the skip block");
    }
    {
        TextConfig b = c; // the first ratio-1 block is no kv source
        b.kv_source[20] = false;
        failures += check(prompt_skip_layer(b) == -1, "a skip block that is no kv source");
    }
    {
        TextConfig b = c; // no ratio-1 block at all
        for (auto& r : b.compress_ratios) { r = r == 1 ? 2 : r; }
        failures += check(prompt_skip_layer(b) == -1, "no ratio-1 block");
    }
    {
        TextConfig b = c; // one block
        b.num_hidden_layers = 1;
        b.compress_ratios.resize(1);
        b.kv_source.resize(1);
        b.index_source.resize(1);
        failures += check(prompt_skip_layer(b) == -1, "a one-block model");
    }
    {
        TextConfig b = c; // vectors of another length than the block count
        b.kv_source.pop_back();
        failures += check(prompt_skip_layer(b) == -1, "a config with a short kv_source");
    }
    return failures;
}

int test_plan_skip() {
    int failures = 0;
    constexpr std::uint32_t keep = 2668;
    // A run of exactly keep rows (or fewer) keeps every row.
    failures += check(plan_prompt_skip(2048, 2048, keep) == 0, "a short run skipped rows");
    failures += check(plan_prompt_skip(keep, keep, keep) == 0, "a run of keep rows skipped rows");
    // Fewer than kPromptSkipMinRows rows to skip: none.
    failures += check(plan_prompt_skip(3000, 3000, keep) == 0,
                      "332 skippable rows are below the minimum yet were skipped");
    failures += check(plan_prompt_skip(4000, 4000, keep) == 4000 - keep, "1332 skippable rows");
    failures += check(plan_prompt_skip(keep + 1023, keep + 1023, keep) == 0, "1023 rows skipped");
    failures += check(plan_prompt_skip(keep + 1024, keep + 1024, keep) == 1024, "1024 rows not skipped");
    // The end of a prompt run in one step: all but the last keep rows.
    failures += check(plan_prompt_skip(9792, 9792, keep) == 9792 - keep, "9.8K prompt's skip");
    // A step that ends before the kept rows is skipped whole.
    failures += check(plan_prompt_skip(16384, 40000, keep) == 16384, "a step before the kept rows");
    // A step that contains the cut skips up to it ...
    failures += check(plan_prompt_skip(16384, 16384 + keep + 5000, keep) == 16384,
                      "a step ending 5000 rows before the cut");
    failures += check(plan_prompt_skip(16384, 16384 + 1000, keep) == 16384 + 1000 - keep,
                      "a step containing the cut");
    // ... but keeps at least kPromptSkipMinKept rows when it keeps any.
    failures += check(plan_prompt_skip(16384, 16384 + keep - 100, keep) ==
                          16384 - kPromptSkipMinKept,
                      "a step with 100 rows past the cut keeps fewer than the minimum");
    failures += check(plan_prompt_skip(16384, 16384 + keep - 512, keep) == 16384 - 512,
                      "a step with exactly the minimum past the cut");
    failures += check(plan_prompt_skip(16384, 16384 + keep - 513, keep) == 16384 - 513,
                      "a step with one row more than the minimum past the cut");
    // A step shorter than the minimum kept rows never keeps a tail of fewer.
    failures += check(plan_prompt_skip(1500, keep + 700, keep) == 0,
                      "a 1500-row step with only 700 rows to skip");
    // A frontier closer than the step cannot be (the caller never cuts past it).
    failures += check(plan_prompt_skip(5000, 4000, keep) == 0, "a frontier inside the step");
    // The last row is always kept at the end of a run, whatever the keep.
    for (std::uint32_t n = 3000; n < 70000; n += 977) {
        const std::uint32_t skip = plan_prompt_skip(n, n, keep);
        failures += check(skip == 0 || n - skip >= keep, "a run's end keeps fewer than keep rows");
        failures += check(skip == 0 || skip >= kPromptSkipMinRows, "a skip below the minimum");
    }
    return failures;
}

int test_pieces() {
    int failures = 0;
    std::vector<SkipPiece> pieces;
    // No skip: the plain split from row 0, every block.
    for (const int layer : {0, 19, 20, 21, 39}) {
        plan_skip_pieces(layer, 20, 5000, 2048, 0, pieces);
        failures += check(pieces.size() == 3 && pieces[0].row == 0 && pieces[0].rows == 2048 &&
                              pieces[1].row == 2048 && pieces[1].rows == 2048 &&
                              pieces[2].row == 4096 && pieces[2].rows == 904 &&
                              !pieces[0].kv_only && !pieces[1].kv_only && !pieces[2].kv_only,
                          "the plain split with no skip");
    }
    plan_skip_pieces(25, -1, 5000, 2048, 3000, pieces);
    failures += check(pieces.size() == 3, "a skip with no skip block");
    // A skip of 3000 of 5000 rows, chunks of 2048, skip block 20.
    plan_skip_pieces(19, 20, 5000, 2048, 3000, pieces);
    failures += check(pieces.size() == 3 && !pieces[0].kv_only && pieces[2].row == 4096,
                      "blocks before the skip block keep the plain split");
    plan_skip_pieces(20, 20, 5000, 2048, 3000, pieces);
    failures += check(pieces.size() == 3 && pieces[0].row == 0 && pieces[0].rows == 2048 &&
                          pieces[0].kv_only && pieces[1].row == 2048 && pieces[1].rows == 952 &&
                          pieces[1].kv_only && pieces[2].row == 3000 && pieces[2].rows == 2000 &&
                          !pieces[2].kv_only,
                      "the skip block's kv_only pieces and its pieces from the cut");
    plan_skip_pieces(21, 20, 5000, 2048, 3000, pieces);
    failures += check(pieces.size() == 1 && pieces[0].row == 3000 && pieces[0].rows == 2000 &&
                          !pieces[0].kv_only,
                      "a block past the skip block runs the rows from the cut");
    // A cut on a chunk boundary and a pass skipped whole.
    plan_skip_pieces(20, 20, 6000, 2048, 4096, pieces);
    failures += check(pieces.size() == 3 && pieces[0].kv_only && pieces[1].kv_only &&
                          pieces[2].row == 4096 && pieces[2].rows == 1904 && !pieces[2].kv_only,
                      "a cut on a chunk boundary");
    plan_skip_pieces(30, 20, 6000, 2048, 6000, pieces);
    failures += check(pieces.empty(), "a pass skipped whole runs a block past the skip block");
    plan_skip_pieces(20, 20, 6000, 2048, 6000, pieces);
    failures += check(pieces.size() == 3 && pieces[0].kv_only && pieces[1].kv_only &&
                          pieces[2].kv_only && pieces[2].rows == 6000 - 4096,
                      "a pass skipped whole runs only kv_only at the skip block");

    // Every row of every block is covered once or skipped, in order, by pieces of at most `chunk`
    // rows; the kv_only pieces are the skip block's rows below the cut.
    std::mt19937 gen(7);
    for (int trial = 0; trial < 2000; ++trial) {
        const int chunk = 128 * (1 + int(gen() % 32));
        const int n     = 1 + int(gen() % 40000);
        const int skip  = int(gen() % std::uint32_t(n + 1));
        const int at    = 1 + int(gen() % 30);
        for (const int layer : {0, at - 1, at, at + 1, at + 7}) {
            plan_skip_pieces(layer, at, n, chunk, skip, pieces);
            int next_full = layer < at ? 0 : skip;
            int next_kv   = 0;
            bool ok       = true;
            for (const SkipPiece& p : pieces) {
                ok = ok && p.rows > 0 && p.rows <= chunk && p.row + p.rows <= n;
                if (p.kv_only) {
                    ok = ok && layer == at && p.row == next_kv && p.row + p.rows <= skip;
                    next_kv += p.rows;
                } else {
                    ok = ok && p.row == next_full;
                    next_full += p.rows;
                }
            }
            ok = ok && next_full == n && (layer != at || next_kv == skip);
            failures += check(ok, "the pieces of a block do not cover its rows");
        }
    }

    // Row-layers of work, as Strata counts them: n * 20 + keep * 20 for a 9792-token prompt.
    const int n = 9792, skip = n - 2668;
    long long work = 0;
    for (int layer = 0; layer < 40; ++layer) {
        plan_skip_pieces(layer, 20, n, 2048, skip, pieces);
        for (const SkipPiece& p : pieces) { work += p.kv_only ? 0 : p.rows; }
    }
    failures += check(work == 249200, "the row-layers of a 9.8K prompt with the skip are not 249200");
    failures += check(40LL * n == 391680, "the row-layers of a 9.8K prompt without the skip");
    return failures;
}

// The exactness argument as a dependency model. latent[l][r]: block l's latent of row r is that of
// the full pass. out[l][r]: block l's output at row r is. Block l < skip_at runs every row; at
// skip_at and past it, rows from the cut: a row's output needs the latents of its window (the
// window - 1 rows before it and itself), a row's latent needs the block's input (the block before
// it's output) at the row. The ring a checkpoint keeps and the logits need: each block's latents of
// the last `window` rows and the last block's output at the last row.
bool exact_after(int layers, int skip_at, int window, int n, int skip) {
    std::vector<std::vector<char>> out(std::size_t(layers), std::vector<char>(std::size_t(n), 0));
    std::vector<std::vector<char>> latent = out;
    for (int l = 0; l < layers; ++l) {
        for (int r = 0; r < n; ++r) {
            const bool computed = l < skip_at || r >= skip;
            const bool input    = l == 0 || out[std::size_t(l - 1)][std::size_t(r)] != 0;
            latent[std::size_t(l)][std::size_t(r)] = computed && input;
        }
        for (int r = 0; r < n; ++r) {
            const bool computed = l < skip_at || r >= skip;
            bool windowed       = computed;
            for (int w = std::max(0, r - window + 1); windowed && w <= r; ++w) {
                windowed = latent[std::size_t(l)][std::size_t(w)] != 0;
            }
            out[std::size_t(l)][std::size_t(r)] = windowed;
        }
    }
    for (int l = 0; l < layers; ++l) {
        for (int r = n - window; r < n; ++r) {
            if (r >= 0 && !latent[std::size_t(l)][std::size_t(r)]) { return false; }
        }
    }
    return out[std::size_t(layers - 1)][std::size_t(n - 1)] != 0;
}

int test_exactness_model() {
    int failures = 0;
    std::mt19937 gen(11);
    for (int trial = 0; trial < 300; ++trial) {
        const int layers = 4 + int(gen() % 9);
        const int at     = 1 + int(gen() % std::uint32_t(layers - 2));
        const int window = 2 + int(gen() % 12);
        const auto keep  = int(prompt_skip_keep(at, std::uint32_t(layers), std::uint32_t(window)));
        // Exactly what the argument needs: the damage of (layers - at - 1) blocks, then a window.
        const int need = (layers - at - 1) * (window - 1) + window;
        const int n    = keep + 50 + int(gen() % 400);
        failures += check(keep == need + window - 1, "keep is not the needed rows and a window's worth");
        failures += check(exact_after(layers, at, window, n, n - keep),
                          "the pass with keep kept rows is not exact");
        failures += check(exact_after(layers, at, window, n, n - need),
                          "the pass with the needed rows kept is not exact");
        failures += check(!exact_after(layers, at, window, n, n - need + 1),
                          "one row fewer than needed is still exact: the model or the bound is off");
    }
    // The release: 40 blocks, skip block 20, a window of 128: 2541 rows needed, 2668 kept.
    failures += check(exact_after(40, 20, 128, 4000, 4000 - 2668) &&
                          exact_after(40, 20, 128, 4000, 4000 - 2541) &&
                          !exact_after(40, 20, 128, 4000, 4000 - 2540),
                      "the release's bound is not 2541 rows with 2668 kept");
    return failures;
}

int test_chunk_candidates() {
    int failures = 0;
    using V = std::vector<std::uint32_t>;
    failures += check(prompt_chunk_candidates(0) == V({2048, 1024, 512}), "no request");
    failures += check(prompt_chunk_candidates(4096) == V({4096, 2048, 1024, 512}), "4096");
    failures += check(prompt_chunk_candidates(8192) == V({4096, 2048, 1024, 512}), "8192 is clamped");
    failures += check(prompt_chunk_candidates(2048) == V({2048, 1024, 512}), "2048");
    failures += check(prompt_chunk_candidates(3072) == V({3072, 2048, 1024, 512}), "3072");
    failures += check(prompt_chunk_candidates(1024) == V({1024, 512}), "1024");
    failures += check(prompt_chunk_candidates(512) == V({512}), "512");
    failures += check(prompt_chunk_candidates(256) == V({256}), "256");
    failures += check(prompt_chunk_candidates(128) == V({128}), "128");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_layer_and_keep();
    failures += test_plan_skip();
    failures += test_pieces();
    failures += test_exactness_model();
    failures += test_chunk_candidates();
    if (failures == 0) { std::cout << "prompt skip: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
