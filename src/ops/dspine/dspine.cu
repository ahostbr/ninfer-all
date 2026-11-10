// DSpine code bridge and greedy serial refinement (see include/ninfer/ops/dspine.h). Rounding points
// follow github.com/Tencent/DSpine dspine_mlx/bridge.py and propose.py; GEMMs are cuBLAS BF16 with
// an FP32 accumulator and a BF16 result, as the reference's BF16 matmuls are.
#include "ninfer/ops/dspine.h"

#include "core/device.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::dspine {
namespace {

constexpr float kF32Tiny = 1.1754943508222875e-38f; // torch.finfo(float32).tiny, the RMS floor
constexpr float kSqrtCode = 32.0f;                   // sqrt(1024): exact in BF16

cublasHandle_t handle(cudaStream_t stream) {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= int(handles.size())) throw std::invalid_argument("dspine: device");
    std::call_once(once[device], [&] {
        if (cublasCreate(&handles[device]) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("dspine: cublasCreate failed");
    });
    if (cublasSetStream(handles[device], stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("dspine: cublasSetStream failed");
    return handles[device];
}

// out[m, n] (ld m) = op(A)[m, k] * B[k, n] (ld k), A given row-major [m, k] (= column-major k x m).
void gemm_rowmajor_a(const void* a_rowmajor, const void* b, void* out, int m, int n, int k,
                     float alpha, cudaStream_t stream) {
    const float beta = 0.0f;
    const auto status = cublasGemmEx(handle(stream), CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                                     a_rowmajor, CUDA_R_16BF, k, b, CUDA_R_16BF, k, &beta, out,
                                     CUDA_R_16BF, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("dspine: cublasGemmEx failed");
}

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

template <int Threads>
__device__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < Threads / 32; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// rms over the summed layer output; normalized = BF16(f / (rms + eps)); optional raw/rms stash.
__global__ void rms_normalize_kernel(const __nv_bfloat16* __restrict__ residual, int hidden,
                                     float eps, __nv_bfloat16* __restrict__ normalized,
                                     float* __restrict__ rms_out,
                                     __nv_bfloat16* __restrict__ stash_raw,
                                     float* __restrict__ stash_rms) {
    __shared__ float scratch[32];
    const int col  = blockIdx.x;
    const auto* x  = residual + static_cast<std::int64_t>(hidden) * col;
    float sum      = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        const float f = bf(x[h]);
        sum += f * f;
    }
    sum             = block_sum<256>(sum, scratch);
    const float rms = sqrtf(fmaxf(sum / static_cast<float>(hidden), kF32Tiny));
    auto* out       = normalized + static_cast<std::int64_t>(hidden) * col;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = __float2bfloat16(bf(x[h]) / (rms + eps));
        if (stash_raw) stash_raw[static_cast<std::int64_t>(hidden) * col + h] = x[h];
    }
    if (threadIdx.x == 0) {
        rms_out[col] = rms;
        if (stash_rms) stash_rms[col] = rms;
    }
}

// codes = BF16(32 * p / (||p|| + eps)); the anchor row takes BF16(32 * codebook[anchor]).
__global__ void codes_kernel(const __nv_bfloat16* __restrict__ projected,
                             const __nv_bfloat16* __restrict__ codebook,
                             const std::int32_t* __restrict__ ids, int width, int code, float eps,
                             __nv_bfloat16* __restrict__ codes) {
    __shared__ float scratch[32];
    const int col = blockIdx.x;
    const int t   = col % width;
    auto* out     = codes + static_cast<std::int64_t>(code) * col;
    if (t == 0) {
        const auto* row = codebook + static_cast<std::int64_t>(code) * ids[col];
        for (int c = threadIdx.x; c < code; c += blockDim.x)
            out[c] = __float2bfloat16(kSqrtCode * bf(row[c]));
        return;
    }
    const auto* p = projected + static_cast<std::int64_t>(code) * col;
    float sum     = 0.0f;
    for (int c = threadIdx.x; c < code; c += blockDim.x) {
        const float f = bf(p[c]);
        sum += f * f;
    }
    sum              = block_sum<256>(sum, scratch);
    const float norm = sqrtf(sum);
    for (int c = threadIdx.x; c < code; c += blockDim.x)
        out[c] = __float2bfloat16(kSqrtCode * bf(p[c]) / (norm + eps));
}

// message_t = BF16(sigmoid(q_t + s_{t-1} + b) * v_{t-1}); zero on the anchor row. Stashes q.
__global__ void message_kernel(const __nv_bfloat16* __restrict__ q,
                               const __nv_bfloat16* __restrict__ s,
                               const __nv_bfloat16* __restrict__ v,
                               const __nv_bfloat16* __restrict__ bias, int width, int rank,
                               __nv_bfloat16* __restrict__ message,
                               __nv_bfloat16* __restrict__ stash_q) {
    const int col = blockIdx.x;
    const int t   = col % width;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        const std::int64_t here = static_cast<std::int64_t>(rank) * col + r;
        if (stash_q) stash_q[here] = q[here];
        if (t == 0) {
            message[here] = __float2bfloat16(0.0f);
            continue;
        }
        const std::int64_t before = here - rank;
        const float z             = bf(q[here]) + bf(s[before]) + bf(bias[r]);
        const float gate          = 1.0f / (1.0f + expf(-z));
        message[here]             = __float2bfloat16(gate * bf(v[before]));
    }
}

// out = BF16(f + rms * delta) on every non-anchor row.
__global__ void apply_kernel(__nv_bfloat16* __restrict__ residual,
                             const __nv_bfloat16* __restrict__ delta,
                             const float* __restrict__ rms, int hidden, int width) {
    const int col = blockIdx.x;
    if (col % width == 0) return;
    const float scale = rms[col];
    auto* x           = residual + static_cast<std::int64_t>(hidden) * col;
    const auto* d     = delta + static_cast<std::int64_t>(hidden) * col;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        x[h] = __float2bfloat16(bf(x[h]) + scale * bf(d[h]));
}

__global__ void scale_by_norm_kernel(const __nv_bfloat16* __restrict__ raw,
                                     const __nv_bfloat16* __restrict__ norm, int hidden,
                                     __nv_bfloat16* __restrict__ out) {
    const std::int64_t base = static_cast<std::int64_t>(hidden) * blockIdx.x;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        out[base + h] = __float2bfloat16(bf(raw[base + h]) * bf(norm[h]));
}

__global__ void tg_input_kernel(const __nv_bfloat16* __restrict__ write_last,
                                const __nv_bfloat16* __restrict__ norm, int hidden, int rank,
                                __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        out[h + static_cast<std::int64_t>(hidden) * r] =
            __float2bfloat16(bf(norm[h]) * bf(write_last[static_cast<std::int64_t>(h) * rank + r]));
}

__device__ __forceinline__ float load_score(const void* p, bool f32, std::int64_t i) {
    return f32 ? static_cast<const float*>(p)[i] : bf(static_cast<const __nv_bfloat16*>(p)[i]);
}

__global__ void transpose_kernel(const void* __restrict__ in, bool f32, int vocab, int rank,
                                 __nv_bfloat16* __restrict__ out) {
    __shared__ float tile[32][33];
    const int v0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int v = v0 + threadIdx.x, r = r0 + i;
        tile[i][threadIdx.x] =
            (v < vocab && r < rank) ? load_score(in, f32, v + static_cast<std::int64_t>(vocab) * r)
                                    : 0.0f;
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int r = r0 + threadIdx.x, v = v0 + i;
        if (v < vocab && r < rank)
            out[r + static_cast<std::int64_t>(rank) * v] = __float2bfloat16(tile[threadIdx.x][i]);
    }
}

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

// One CTA per column: per-thread sorted top-16 over a strided slice, then 16 block-wide
// extractions (stable: lower id wins a tie).
__global__ void topk_kernel(const void* __restrict__ logits, bool f32, std::int64_t ld,
                            int valid_rows, int k, std::int32_t* __restrict__ ids) {
    constexpr int K = 16;
    float best[K];
    int best_id[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        best[i]    = -INFINITY;
        best_id[i] = 0x7fffffff;
    }
    const std::int64_t base = ld * blockIdx.x;
    for (int v = threadIdx.x; v < valid_rows; v += blockDim.x) {
        const float s = load_score(logits, f32, base + v);
        if (!better(s, v, best[K - 1], best_id[K - 1])) continue;
        int pos = K - 1;
        while (pos > 0 && better(s, v, best[pos - 1], best_id[pos - 1])) {
            best[pos]    = best[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best[pos]    = s;
        best_id[pos] = v;
    }
    __shared__ float s_score[256];
    __shared__ int s_id[256];
    __shared__ int s_owner[256];
    int head = 0;
    for (int round = 0; round < k; ++round) {
        s_score[threadIdx.x] = head < K ? best[head] : -INFINITY;
        s_id[threadIdx.x]    = head < K ? best_id[head] : 0x7fffffff;
        s_owner[threadIdx.x] = threadIdx.x;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const int o = threadIdx.x + stride;
                if (better(s_score[o], s_id[o], s_score[threadIdx.x], s_id[threadIdx.x])) {
                    s_score[threadIdx.x] = s_score[o];
                    s_id[threadIdx.x]    = s_id[o];
                    s_owner[threadIdx.x] = s_owner[o];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) ids[static_cast<std::int64_t>(k) * blockIdx.x + round] = s_id[0];
        const int owner = s_owner[0];
        __syncthreads();
        if (threadIdx.x == owner) {
            // Registers cannot be indexed by a runtime value without spilling; shift instead.
#pragma unroll
            for (int i = 0; i < K - 1; ++i) {
                best[i]    = best[i + 1];
                best_id[i] = best_id[i + 1];
            }
            best[K - 1]    = -INFINITY;
            best_id[K - 1] = 0x7fffffff;
        }
        __syncthreads();
    }
    (void)head;
}

// One CTA per batch row; rows n = 0..k-1 are proposals t = n + 1. Dynamic smem: m [16, R] FP32.
__global__ void refine_kernel(const std::int32_t* __restrict__ cand, const void* __restrict__ base,
                              bool base_f32, std::int64_t base_ld,
                              const __nv_bfloat16* __restrict__ stash_q,
                              const float* __restrict__ stash_rms,
                              const std::int32_t* __restrict__ ids, int width,
                              const __nv_bfloat16* __restrict__ tg,
                              const __nv_bfloat16* __restrict__ tsv_gate,
                              const __nv_bfloat16* __restrict__ tsv_value,
                              const __nv_bfloat16* __restrict__ bias, int rank, int top_k,
                              std::int32_t* __restrict__ drafts) {
    constexpr int K = 16;
    extern __shared__ float m[]; // [K][rank]
    __shared__ int cand_all[16][K];
    __shared__ int idx_all[16][K];
    __shared__ int sender[K];
    __shared__ float score[K][K];
    __shared__ float base_v[K];
    const int b      = blockIdx.x;
    const int k      = width - 1;
    const int anchor = ids[static_cast<std::int64_t>(width) * b];
    for (int n = 0; n < k; ++n) {
        const int col  = n + k * b;          // candidate / base column
        const int row  = (n + 1) + width * b; // block column of the stash
        if (threadIdx.x < top_k) {
            const int c                  = cand[static_cast<std::int64_t>(top_k) * col + threadIdx.x];
            cand_all[n][threadIdx.x]     = c;
            sender[threadIdx.x]          = n == 0 ? anchor : cand_all[n - 1][threadIdx.x];
            base_v[threadIdx.x]          = load_score(base, base_f32, base_ld * col + c);
        }
        __syncthreads();
        const float rms = stash_rms[row];
        for (int i = threadIdx.x; i < top_k * rank; i += blockDim.x) {
            const int a = i / rank, r = i % rank;
            const std::int64_t at = r + static_cast<std::int64_t>(rank) * sender[a];
            const float z = bf(stash_q[static_cast<std::int64_t>(rank) * row + r]) + bf(tsv_gate[at]) +
                            bf(bias[r]);
            m[a * rank + r] = (1.0f / (1.0f + expf(-z))) * bf(tsv_value[at]);
        }
        __syncthreads();
        for (int p = threadIdx.x; p < top_k * top_k; p += blockDim.x) {
            const int a = p / top_k, j = p % top_k;
            const auto* t = tg + static_cast<std::int64_t>(rank) * cand_all[n][j];
            float dot     = 0.0f;
            for (int r = 0; r < rank; ++r) dot += m[a * rank + r] * bf(t[r]);
            score[a][j] = base_v[j] + rms * dot;
        }
        __syncthreads();
        if (threadIdx.x < top_k) {
            int arg = 0;
            for (int j = 1; j < top_k; ++j)
                if (score[threadIdx.x][j] > score[threadIdx.x][arg]) arg = j;
            idx_all[n][threadIdx.x] = arg;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int j = idx_all[0][0];
        for (int n = 0; n < k; ++n) {
            drafts[n + k * b] = cand_all[n][j];
            if (n + 1 < k) j = idx_all[n + 1][j];
        }
    }
}

const __nv_bfloat16* as_bf16(const void* p) { return static_cast<const __nv_bfloat16*>(p); }

} // namespace

void bridge_write(Tensor& residual, const Tensor& ids, std::int32_t width, std::int32_t batch,
                  const BridgeWeights& w, std::int32_t layer, BridgeStash* stash,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    if (residual.dtype != DType::BF16 || ids.dtype != DType::I32 || w.code != 1024 ||
        layer < 0 || layer >= w.layers || width < 2 || width > 16)
        throw std::invalid_argument("dspine bridge_write: unsupported shape");
    auto scope       = workspace.scope();
    const int cols   = width * batch;
    const int H      = w.hidden, C = w.code, R = w.rank;
    Tensor normalized = workspace.alloc(DType::BF16, {H, cols});
    Tensor rms        = workspace.alloc(DType::FP32, {cols});
    Tensor projected  = workspace.alloc(DType::BF16, {C, cols});
    Tensor codes      = workspace.alloc(DType::BF16, {C, cols});
    Tensor q          = workspace.alloc(DType::BF16, {R, cols});
    Tensor s          = workspace.alloc(DType::BF16, {R, cols});
    Tensor v          = workspace.alloc(DType::BF16, {R, cols});
    Tensor message    = workspace.alloc(DType::BF16, {R, cols});
    Tensor delta      = workspace.alloc(DType::BF16, {H, cols});
    auto* raw_out     = stash ? static_cast<__nv_bfloat16*>(stash->raw.data) : nullptr;
    auto* rms_stash   = stash ? static_cast<float*>(stash->rms.data) : nullptr;
    rms_normalize_kernel<<<cols, 256, 0, stream>>>(
        as_bf16(residual.data), H, w.eps, static_cast<__nv_bfloat16*>(normalized.data),
        static_cast<float*>(rms.data), raw_out, rms_stash);
    CUDA_CHECK(cudaGetLastError());
    const auto* read = layer == w.layers - 1
                           ? as_bf16(w.read_weight)
                           : as_bf16(w.early_read_weights) + static_cast<std::int64_t>(layer) * C * H;
    gemm_rowmajor_a(read, normalized.data, projected.data, C, cols, H, 1.0f, stream);
    codes_kernel<<<cols, 256, 0, stream>>>(as_bf16(projected.data), as_bf16(w.codebook),
                                           static_cast<const std::int32_t*>(ids.data), width, C,
                                           w.eps, static_cast<__nv_bfloat16*>(codes.data));
    CUDA_CHECK(cudaGetLastError());
    gemm_rowmajor_a(w.gate_weight, codes.data, q.data, R, cols, C, 1.0f, stream);
    gemm_rowmajor_a(w.predecessor_gate_weight, codes.data, s.data, R, cols, C, 1.0f, stream);
    gemm_rowmajor_a(w.prev_weight, codes.data, v.data, R, cols, C, 1.0f, stream);
    message_kernel<<<cols, 256, 0, stream>>>(
        as_bf16(q.data), as_bf16(s.data), as_bf16(v.data), as_bf16(w.gate_bias), width, R,
        static_cast<__nv_bfloat16*>(message.data),
        stash ? static_cast<__nv_bfloat16*>(stash->q.data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
    const auto* write =
        as_bf16(w.feature_write_weights) + static_cast<std::int64_t>(layer) * H * R;
    gemm_rowmajor_a(write, message.data, delta.data, H, cols, R, 1.0f, stream);
    apply_kernel<<<cols, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(residual.data),
                                           as_bf16(delta.data), static_cast<const float*>(rms.data),
                                           H, width);
    CUDA_CHECK(cudaGetLastError());
}

void build_tsv(const BridgeWeights& w, Tensor& tsv_gate, Tensor& tsv_value, cudaStream_t stream) {
    // TSV[v] = BF16(32 * codebook[v]) @ W^T; 32 is a power of two, so alpha = 32 equals scaling the
    // BF16 codebook first.
    gemm_rowmajor_a(w.predecessor_gate_weight, w.codebook, tsv_gate.data, w.rank, w.vocab, w.code,
                    kSqrtCode, stream);
    gemm_rowmajor_a(w.prev_weight, w.codebook, tsv_value.data, w.rank, w.vocab, w.code, kSqrtCode,
                    stream);
}

void transpose_to_rank_major(const Tensor& vocab_major, Tensor& rank_major, cudaStream_t stream) {
    const int vocab = vocab_major.ne[0], rank = vocab_major.ne[1];
    const dim3 grid((vocab + 31) / 32, (rank + 31) / 32);
    transpose_kernel<<<grid, dim3(32, 8), 0, stream>>>(
        vocab_major.data, vocab_major.dtype == DType::FP32, vocab, rank,
        static_cast<__nv_bfloat16*>(rank_major.data));
    CUDA_CHECK(cudaGetLastError());
}

void tg_input(const BridgeWeights& w, const Tensor& norm, Tensor& out, cudaStream_t stream) {
    const auto* last = as_bf16(w.feature_write_weights) +
                       static_cast<std::int64_t>(w.layers - 1) * w.hidden * w.rank;
    tg_input_kernel<<<w.rank, 256, 0, stream>>>(last, as_bf16(norm.data), w.hidden, w.rank,
                                                static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void scale_by_norm(const Tensor& raw, const Tensor& norm, Tensor& out, cudaStream_t stream) {
    const int hidden = raw.ne[0], cols = raw.ne[1];
    scale_by_norm_kernel<<<cols, 256, 0, stream>>>(as_bf16(raw.data), as_bf16(norm.data), hidden,
                                                   static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void topk_columns(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                  cudaStream_t stream) {
    if (k < 1 || k > 16 || valid_rows > logits.ne[0])
        throw std::invalid_argument("dspine topk_columns: k must be 1..16");
    topk_kernel<<<logits.ne[1], 256, 0, stream>>>(logits.data, logits.dtype == DType::FP32,
                                                  logits.ne[0], valid_rows, k,
                                                  static_cast<std::int32_t*>(ids.data));
    CUDA_CHECK(cudaGetLastError());
}

void refine_greedy(const Tensor& cand, const Tensor& base_logits, const BridgeStash& stash,
                   const Tensor& ids, std::int32_t width, std::int32_t batch, const Tensor& tg,
                   const Tensor& tsv_gate, const Tensor& tsv_value, const void* gate_bias,
                   std::int32_t rank, std::int32_t top_k, Tensor& drafts, cudaStream_t stream) {
    if (top_k != 16 || width < 2 || width > 16)
        throw std::invalid_argument("dspine refine_greedy: head_topk must be 16, width 2..16");
    const std::size_t smem = static_cast<std::size_t>(top_k) * rank * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(refine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));
    refine_kernel<<<batch, 256, smem, stream>>>(
        static_cast<const std::int32_t*>(cand.data), base_logits.data,
        base_logits.dtype == DType::FP32, base_logits.ne[0], as_bf16(stash.q.data),
        static_cast<const float*>(stash.rms.data), static_cast<const std::int32_t*>(ids.data),
        width, as_bf16(tg.data), as_bf16(tsv_gate.data), as_bf16(tsv_value.data),
        as_bf16(gate_bias), rank, top_k, static_cast<std::int32_t*>(drafts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::dspine
