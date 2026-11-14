#pragma once

#include "ninfer/model_enums.h"

namespace ninfer::runtime {

// Fork change W5 (docs/fork/WINDOWS_PORT.md): whether a Qwen3.8-Flash-Next / Edge0 engine must run
// without CUDA Graphs. With host or disk experts, a captured graph waits for the experts' host
// service on the GPU (CudaHostHandshake::wait_prepared/wait_completed launch a wait_phase kernel
// inside a capture, core/cuda_host_handshake.cu), while eager execution waits on the host. Where a
// device's copies queue behind one another (serialized_copy_queue(): Windows WDDM, WSL2), the
// graph-side wait hung the warm-up of Edge0-145B-NVFP4 on an RTX 5090 (2026-11-14: "preparing CUDA
// graphs", then no further log line, the GPU at 100%, until the device watchdog ended the process;
// the same start with --no-cuda-graph was ready in 23.5 s). Device experts never wait for the host.
[[nodiscard]] constexpr bool host_waits_need_eager(ExpertResidency experts,
                                                   bool serialized_copies) noexcept {
    return serialized_copies && experts != ExpertResidency::Device;
}

} // namespace ninfer::runtime
