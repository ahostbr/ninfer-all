# DSpine draft port plan (fork, 2026-11-10, UNVERIFIED)

> **Superseded on 2026-11-14.** The upstream sync (merge `c6a560d1`, see
> [SYNC_AUDIT_20261114.md](SYNC_AUDIT_20261114.md) section 6) replaced this port with upstream's own DSpine:
> a separate drafter artifact run with `--spec dspine --draft-model`. An artifact made by this port's
> converter (the drafter inside the `dflash` component) is refused by builds from that merge on
> (`DFlash config: unknown member dspine`). This file records the port as it was.

Ryan 2026-11-10 ~00:5x EST: "Yes do it ourself. Now while he stays on pz" (after "won't it also help the non-nvfp4?" and "the swift and normal 27b?").

## What DSpine is
Tencent's draft for Qwen3.8-27B (`Qwen3.8-27B-DSpine-b16`, Apache-2.0, 4.05 GB BF16, local copy
`C:/Users/Ryan/.litesuite/llm/ninfer-models/hf/Qwen3.8-27B-DSpine-b16`). Architecture string
`DFlashDraftModel`: the DFlash body (5 Qwen3 layers, sliding window 8192, `fc` [5120, 25600] over 5 target
layers `target_layer_ids` [1,16,31,46,61], `hidden_norm`, mask token 248070), block size 16, plus:

1. **Code bridge** (`selected_prefix_bridge.*`), one write after every draft layer (reference
   `dspine_mlx/bridge.py: DSpineCodeBridge.__call__`, mirrors `sglang_dspine/models/dspine.py:_forward_eager`):
   RMS-normalise the layer output, project to a 1024-d code (`early_read_weights[l]`, last layer `read_weight`),
   L2-normalise x sqrt(1024); row 0 of the block takes the anchor token's `codebook` row instead; one
   GEMM against packed [gate; predecessor_gate; prev] (3 x 512 rows); message_t = sigmoid(q_t + s_{t-1} + b) * v_{t-1}
   (row 0 gets 0); delta = message x `feature_write_weights[l]`^T; out = x + rms * delta on non-anchor rows;
   then the next layer's input norm. Exact BF16/FP32 rounding points are in the reference; keep them.
   Load checks: `independent_write_version == 1`, `terminal_disabled_version == 1`, `out.weight` all zero.
2. **Serial refinement tail** (`code_bridge_second_pass: serial_table, first_exact 4, head_topk 16`;
   reference `dspine_mlx/propose.py: RefinementTail`): from the last layer's stash (raw, rms, gate part q),
   top-16 candidates per row from the target LM head, then a 16x16 score table per row
   (greedy: base + rms * (sigmoid(q + TSV_g[prev] + b) * TSV_v[prev]) . TG[cand]) walked as a chain from the anchor.
   TG [V,512] and TSV [V,1024] are precomputed once at load (`build_serial_tables`, needs the target LM head rows
   and the final norm weight). Sampled mode builds the full refined hidden per candidate (`sampled_table`).
3. **Verification:** block 16 instead of DFlash2's 8; Tencent pairs it with DSpark-style dynamic verification
   and ReplaySSM. Both are optional for a first port: plain block verify works with the existing DFlash path.

## Where it lands in NInfer (mirror DFlash2's footprint)
- `src/models/qwen3_5/config.{h,cpp}`: a `dspine` extra in `DraftConfig` (bridge sizes, feature layers, refine params, block 16).
- converter (`tools/`): pack `selected_prefix_bridge.*` under `dspine/...`; keep the bridge BF16 (the reference
  forbids quantising it); the 5 layers and `fc` may take the existing draft quantisation.
- `src/models/qwen3_5/load/dspine.cpp` + `bindings.h` + `loading_sources.cmake`: bind bridge weights; build TG/TSV at load.
- `src/models/qwen3_5/execution/draft.cpp`: a bridge write after each layer (one small CUDA kernel for the
  norm/code/gate/delta elementwise parts around two existing GEMM calls), then the refinement tail.
- `src/product/speculative_options.h`, `apps/cli/options.cpp`, `src/serve/serve_options.cpp`: `--spec dspine`.

## Order, each step proven before the next
1. Golden numbers: run the reference (CPU/torch port of `bridge.py` + `propose.py`, or the MLX code's maths in
   numpy) on one fixed block to dump per-layer `out`, the stash and the greedy chain. That file is the test oracle.
2. Config + converter + loader; a load test that reads every tensor and passes the 3 version checks.
3. Bridge write in the draft forward; test against the golden per-layer outputs (BF16 tolerance).
4. Refinement tail (greedy first); test the chain against the golden ids.
5. End to end: `ninfer-serve --spec dspine` on Qwen3.8-27B; measure accepted tokens per round and tok/s against
   `--spec dflash2` on the same prompts. Then Swift 1.5 (a fine-tune of the same 27B; expect lower acceptance).
6. Later: DSpark-style dynamic verification, ReplaySSM, sampled mode.

## Known risks
- Upstream issue #265: NVFP4 A16 on Blackwell, scale byte 0x7F decodes to NaN. Affects NVFP4 targets only.
- TG/TSV memory: [248320 x 512] + [248320 x 1024] BF16 = about 0.76 GB extra VRAM.
- VRAM rule: any model load for steps 5+ needs `lms ps` empty and Sentinel's approval first.
- This fork's remote (`ahostbr/ninfer-all`) is PUBLIC: commits stay local until Ryan OKs a push.

Reference clone (read-only): `G:/REPO_CLONES/_untrusted/DSpine` (github.com/Tencent/DSpine, 321b0fc).

## Status 2026-11-10 (UNVERIFIED by Ryan)
Ported and running: 1b81a20a..76c35c85 (fork/audited, pushed). Greedy refinement only (sampled mode not ported); no golden-number unit test yet.
Artifact: `G:\Models\litesuite\ninfer\swift15_nvfp4_dspine_p.ninfer` (Swift-1.5 27B NVFP4, `--recipe qwen3_8_27b_nvfp4_nvidia --components text,dflash --proposal --source dflash=<DSpine>`), launch recipe beside it (64K x 2 lanes). Build: `C:\Users\Ryan\.litesuite\llm\ninfer-all-dspine-76c35c85`. LiteTUI points at both (settings.json; backup settings.json.bak-pre-dspine-20261110).
Measured (greedy, 300 tokens, decode tok/s and accepted/round; no draft 74.8): Rust quicksort draft 7 306.2 (5.84), draft 15 401.6 (9.06); Python ISO parser 295.8 (5.75) / 340.7 (7.67); prose 130.7 (2.56) / 113.8 (2.77). Greedy output identical to no-draft at draft 1/3/7; at 15 one near-tie diverges (16-column verification rounding).
Runtime changes beyond the plan: o_proj 5120x4096 run as project + residual_add (no registered linear_add); 8192 cyclic append; a drafter with sliding layers only reserves no paged backend KV; TG/candidates read the target head (the draft head is unbound under --lm-head-draft).
