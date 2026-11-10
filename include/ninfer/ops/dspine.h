#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

// Tencent DSpine draft (github.com/Tencent/DSpine, Apache-2.0): the code bridge written after every
// draft layer and the greedy serial last-write refinement. The arithmetic and its rounding points
// follow dspine_mlx/bridge.py (DSpineCodeBridge.__call__) and dspine_mlx/propose.py
// (RefinementTail.greedy, build_serial_tables). All matrices are column-major NInfer tensors:
// a block of `width` columns per batch row, column t + width * b, row 0 the anchor.
namespace ninfer::ops::dspine {

// HF row-major bridge tensors as converted (dflash/dspine/*), all BF16.
struct BridgeWeights {
    const void* codebook;              // [V, C]
    const void* read_weight;           // [C, H] (last layer)
    const void* early_read_weights;    // [L-1, C, H]
    const void* feature_write_weights; // [L, H, R]
    const void* gate_weight;           // [R, C]
    const void* gate_bias;             // [R]
    const void* predecessor_gate_weight; // [R, C]
    const void* prev_weight;           // [R, C]
    std::int32_t hidden = 0, code = 0, rank = 0, vocab = 0, layers = 0;
    float eps = 1e-6f;
};

// The last layer's pre-write state the refinement reads (dspine_mlx stash): raw BF16 [H, cols],
// its RMS FP32 [cols], and the gate projection q BF16 [R, cols].
struct BridgeStash {
    Tensor raw, rms, q;
};

// One bridge write after draft layer `layer`, in place on `residual` (BF16 [H, width*batch], the
// summed layer output). `ids` is I32 [width, batch]; ids[0, b] is row b's anchor token. With
// `stash`, the pre-write raw/rms/q are copied out first.
void bridge_write(Tensor& residual, const Tensor& ids, std::int32_t width, std::int32_t batch,
                  const BridgeWeights& w, std::int32_t layer, BridgeStash* stash,
                  WorkspaceArena& workspace, cudaStream_t stream);

// TSV gate/value tables, BF16 [R, V] each: column v = 32 * codebook[v] times the predecessor
// gate / prev weights (build_serial_tables' TSV, split, rank fastest).
void build_tsv(const BridgeWeights& w, Tensor& tsv_gate, Tensor& tsv_value, cudaStream_t stream);

// Transposes a [V, R] projection (vocabulary fastest; FP32 or BF16) into BF16 [R, V].
void transpose_to_rank_major(const Tensor& vocab_major, Tensor& rank_major, cudaStream_t stream);

// TG projection input, BF16 [H, R] column-major: column r = BF16(norm[h] * W_last[h, r]), W_last the
// last layer's feature write weights. Projected through the target head it gives TG^T [V, R].
void tg_input(const BridgeWeights& w, const Tensor& norm, Tensor& out, cudaStream_t stream);

// x[:, j] = BF16(raw[:, j] * norm[:]) (the refinement base input).
void scale_by_norm(const Tensor& raw, const Tensor& norm, Tensor& out, cudaStream_t stream);

// Stable top `k` (k <= 16) of each column of `logits` (FP32 or BF16 [V_physical, cols]) over rows
// [0, valid_rows): descending score, lower id first on ties. ids I32 [k, cols].
void topk_columns(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                  cudaStream_t stream);

// Greedy serial refinement for rows t = 1..width-1 of every batch row: per row a [K,K] score table
// base[cand_j] + rms * (sigmoid(q + TSVg[sender] + b) * TSVv[sender]) . TG[cand_j], walked as a
// chain from the anchor. cand I32 [K, k*batch] (column n + k*b, n = t-1), base_logits FP32/BF16
// [V_physical, k*batch], stash rows from the block layout, tg/tsv BF16 [R, V]. drafts I32 [k*batch].
void refine_greedy(const Tensor& cand, const Tensor& base_logits, const BridgeStash& stash,
                   const Tensor& ids, std::int32_t width, std::int32_t batch,
                   const Tensor& tg, const Tensor& tsv_gate, const Tensor& tsv_value,
                   const void* gate_bias, std::int32_t rank, std::int32_t top_k, Tensor& drafts,
                   cudaStream_t stream);

} // namespace ninfer::ops::dspine
