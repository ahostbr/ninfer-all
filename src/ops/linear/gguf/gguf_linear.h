#pragma once

// Products of GGUF block matrices (QType::GGUF_*, QuantLayout::GgufBlocks). Every product quantizes
// its activation to ggml's q8_1 exactly as llama.cpp does -- that is the format's arithmetic, so it
// happens whatever the Use's policy says -- and accumulates in FP32: ggml's vector kernel through
// eight columns, its integer tensor-core kernel above.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

[[nodiscard]] gguf::GgmlType gguf_type(QType format);

// Throws unless `w` is a resident GGUF matrix over whole blocks.
void require_gguf(const Weight& w, const char* label);

enum class GgufEpilogue {
    Store,       // out = W x
    Accumulate,  // out += W x
    GateProduct, // out = silu(gate) * (W x), gate an FP32 plane of the same rows
};

// One GGUF matrix and where its rows land: rows [row, row + weight->n) of `out` (BF16 [R,T]) or,
// with `f32`, of an FP32 plane with `f32_rows` rows per column.
struct GgufProduct {
    const Weight* weight  = nullptr;
    Tensor* out           = nullptr;
    std::int32_t row      = 0;
    GgufEpilogue epilogue = GgufEpilogue::Store;
    const float* gate     = nullptr;
    float* f32            = nullptr;
    std::int32_t f32_rows = 0;
    gguf::GateActivation activation = gguf::GateActivation::Silu; // GateProduct only
};

// What a workspace query needs of a product: its format and [rows, k]. A gather of the input
// (Weight::input_columns) is assumed wherever it can cost bytes.
struct GgufShape {
    QType qtype       = QType::GGUF_Q8_0;
    std::int32_t rows = 0;
    std::int32_t k    = 0;
};

// Transient bytes gguf_project needs for these products (one K) at every width in
// [min_tokens, max_tokens].
//
// `dequantize_wide` (see gguf_project) makes room for the dequantize-and-GEMM route at every width
// past kDequantizedVectorColumns, as for a type without an integer kernel.
[[nodiscard]] std::size_t gguf_project_workspace_bytes(std::span<const GgufShape> shapes,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens,
                                                       bool dequantize_wide = false);

// Runs every product against one BF16 activation x [K,T], quantizing x once per activation layout
// (and once more per distinct input gather).
//
// `dequantize_wide` (--prefill-cublas for DeepSeek-V4.1): past 64 columns every product, whatever
// its type, dequantizes its weight to BF16 a slab of rows at a time and runs a cuBLAS BF16 GEMM
// with an FP32 accumulator, the route a type without an integer kernel takes, in place of the
// integer tensor-core kernel. The activation then stays BF16 (no q8_1 rounding) and the weight
// rounds to BF16: a quality trade measured by perplexity, off by default.
void gguf_project(const Tensor& x, std::span<const GgufProduct> products,
                  WorkspaceArena& workspace, cudaStream_t stream, bool dequantize_wide = false);

void gguf_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream);
void gguf_linear_add(const Tensor& x, const Weight& w, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream);

// out = act(gate x) * (up x), act SiLU unless `activation` says GELU (tanh). `up` null means
// `gate` is one [gate; up] parent of 2 * out rows.
void gguf_swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
                 WorkspaceArena& workspace, cudaStream_t stream,
                 gguf::GateActivation activation = gguf::GateActivation::Silu);
// `up` null: `gate` is the one [gate; up] parent.
[[nodiscard]] std::size_t gguf_swiglu_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens);

// Gather of token-table rows: out[:, i] = W[ids[i], :] as BF16.
void gguf_embedding(const Weight& table, const Tensor& ids, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
