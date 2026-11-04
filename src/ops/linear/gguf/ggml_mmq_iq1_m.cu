// The integer tensor-core product for IQ1_M, in its own unit so the heavy template builds in
// parallel with the others. Only the MoE product of DeepSeek-V4.1 uses it (--deepseek-iq1m-matrix);
// dense IQ1_M products stay on the dequantize-and-GEMM route (has_matrix_kernel).
#include "ggml_bridge_mmq.cuh"

NINFER_GGUF_MATRIX_INSTANCE(GGML_TYPE_IQ1_M);
