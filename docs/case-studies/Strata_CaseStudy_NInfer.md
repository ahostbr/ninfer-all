# Case Study: Strata Patterns for NInfer

## Intent

- Document how Strata (fork branch `fork/merge-0.2.10`) runs the ISTA DeepSeek-V4.1-Flash GSQ-RCO Coder on Ryan's Windows PC, straight from the GGUF shards, as a pattern reference for porting its tricks to NInfer.
- The headline figures Ryan supplied with the request are 46.8 tok/s decode, 41 s load, 31.8 GB VRAM and 77 GB RAM. This study checks them against what is on disk (section 1) and does not restate them as measured here.
- Extract mechanisms (what runs where, when, and why the code chose it), not defects. Scope exclusion: no bug hunt, no review.
- Read-only study. Nothing was built, run or loaded, and no weight file was read. Every line cited below was opened and the number confirmed in this pass.

## Scope and evidence sources

Prefixes are written out in full on purpose; every entry is an absolute path with the line or range that was opened.

### Strata engine, `strata-dsv41` (the C++ and CUDA side)

- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:3-23` - header comment: the stdin/stdout protocol, reuse rules, slot keeping.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:282-335` - the decode "round" (draft path vs plain step); plain path at 330-333.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:390-396` - fatal GPU error: print `ERR`, `_Exit(3)`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:462-475` - `parse_gen`: `GEN <max_new> key=value ...`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:513-520` - `serve()` prints INFO lines then `READY <ctx> stop`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:549-567` - conversation-slot environment knobs and defaults.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1355-1368` - single-conversation request loop; `ERR unsupported command`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1415-1436` - NaN router stop and the `T <id>` token line.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1472-1493` - `DONE` line, per-request stderr stats line, `save_profile()`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1495-1503` - comment for the `--parallel` multi-conversation scheduler.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2536-2540` - `--dspark off|none|auto` parsing.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2602-2608` - `--pack-info` is host-only.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2551` - the `--vision` flag.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2597-2599` - an unknown option prints `unknown option` and exits 2.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2620-2629` - vision lazy default in serve mode; `Model::load`; `serve()`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:26-77` - `Options`: every load-time knob with its comment (profile write-back at 36-39).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:385-400` - `Pack::load` reads `ds4.txt`, takes `model_dir`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:401-434` - `tensors.txt` parser (`file` and `t` lines).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:436-454` - open every shard; for GGUF map read-only plus direct and parallel handles.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:456-481` - `engram.bin` (`DSE1` header, hash tables read into RAM).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1085` - chunked prompt is the default.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1234` - `kRamMargin` = 8 GiB.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1316-1323` - `sync_decode` default, `async_max_t` = 8 (non-Metal build at 1318).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1349-1355` - cache swap tunables.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1965-1985` - `dalloc`: `cudaMalloc` + zero + track.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2034-2043` - `halloc`: `cudaHostAlloc`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2654-2672` - geometry gates and "no CUDA device".
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2705-2711` - tensor-parallel width reduced to what the expert blocks allow.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2754` - a GGUF pack keeps its experts "in the files".
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2777-2800` - prompt chunk size choice and recent-slot count.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2814` - vision tower loaded before the layer split.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2849-2850` - `n_mtp` is zeroed when the drafter is not loaded (`--dspark none`, not `off`); block cap.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3058-3103` - dense weights must fit VRAM; fail or warn.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3194-3216` - per-device scratch through `dalloc`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3252-3254` - compressed-KV and index-key caches per KV source.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3304-3310` - one thread per device runs `load_dense`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3339-3353` - embedding table decoded row by row into pinned host memory.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3394-3397` - pinned host buffers for the CPU-lane exchange and Engram rows.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3410-3422` - Engram row cache sizing and reader paths.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3424-3446` - Engram direct-I/O opt-in and reader start.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3585-3590` - the "read in place ... prompts in chunks of %d" startup line (log line 5).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3993-4017` - `chunk_default`: may raise the chunk to `kMaxChunk` (4 x `kVlRows` = 4 x 8192 = 32768, `:1090` and `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cuh:102`) when the experts stay in the files.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2386-2392` - expert blob layout needs w1/w3/w2 per expert.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9561-9569` - CPU lane orders experts by a lookahead flag.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14130-14139` - lookahead gates for the next layers.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:70-72` - vision lazy comment (about 320 MiB).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3490-3500` - expert-profile read with size check.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3523-3530` - prompt buffers: shrink the chunk when allocation fails.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3757-3760` - CPU-lane service thread, pinned to a core.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3778` - cache swap threads started.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5372` - `place_experts`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5446-5475` - per-layer VRAM expert counts from the budget.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5583-5586` - profile becomes the starting heat.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5834-5883` - `copy_experts`: all-or-none RAM copy and its messages.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:6465-6485` - `cpu_lane_fits_ram`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:6486-6508` - `store_gb` and `auto_copy_all`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7694-7697` - `load_vision`; no tower means text only.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7733` - lazy vision borrows from `--reserve-mib`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7819-7899` - `eg_issue`: Engram n-gram rows, cache, file reads.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7930-7938` - a failed Engram row becomes zeros.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:8761` - `router_nonfinite`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:8802-8832` - `cache_tick`, `store_watch`, heat decay.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9254-9300` - `cache_queue`: hottest in, coldest out.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9387-9393` - `serve_lane` polls the request ring.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9573-9598` - CPU lane computes, writes the response sequence number.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9716-9720` - PCIe streaming lane is opt-in (`DS4_STREAM`).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9753-9754` - `lane_alloc` returns early when the lane is off.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:11375-11380` - `step` is `forward` of one token.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:12453-12457` - `pick_flow_min` startup line.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13230-13234` - `plan_window` decides async vs host-routed.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13359-13410` - `forward`: checks, Engram issue, plan, run.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13441-13449` - `forward_serial` embeds then walks layers.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13999-14001` - Engram rows collected right before their layer.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14122-14146` - async MoE: gate, lookahead, route-and-publish, shared expert.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14176-14196` - GPU experts, wait for the CPU lane, down projection.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14248` - `hc_post` writes the residual.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14453-14456` - `sample` on the head device.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14802-14812` - `save_profile`: temp file then replace.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14816-14822` - the `INFO` line (`kv=fp4`, expert counts).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1607-1625` - `route_publish_kernel` scoring.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1629-1651` - top-k selection, tie rule and non-finite fallback.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1652-1673` - normalization, plan pointer, mailbox publish, conditional host copy.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:43-51` - experts per word rule.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:53-63` - profile size check.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:65-97` - prompt chunk rule with its measurements.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:99-106` - `copy_all_fits`.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:190-196` - copy pin default.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_engram_io.hpp:3-10` - Engram reader design comment.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_store.hpp:1-16` - RAM expert store (the demand store, off in this run).
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cuh:18` - `kWin` = 16, the most tokens a window takes.
- `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_pick_rule.hpp:18` - `kPromptFlowMin` = 96.

### Strata server and build

- `F:\REPO_CLONES\Strata-fork\serve\server.py:133-140` - `end_leftover_engines`.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:217-219` - DeepSeek thinking-budget constants.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:4826` - the default budget applied when none is given.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:5898-5905` - a budget larger than `max_tokens` minus the answer reserve is cut and "thinking capped at ..." is printed.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:378` - default `ENGINE_READY_S` of 900.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:1145-1152` - engine spawn with pipes.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:1176-1184` - INFO collection and READY parsing.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:1228-1240` - `EngineDied` on a start that fails.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:1545-1560` - sampling keys sent on the GEN line.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:2094` and `:2512-2513` - GEN/GENI line construction.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:2821-2840` - engine shutdown ladder.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:3417-3425` - `engine_ready_s` config override.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:3945-3959` - DeepSeek slot environment defaults.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:4053-4057` - `child_env` applies it for `deepseek41`.
- `F:\REPO_CLONES\Strata-fork\serve\server.py:4953-4979` - fatal-GPU restart budget.
- `F:\REPO_CLONES\Strata-fork\CMakeLists.txt:11` and `:80` - project version (reads 0.1.56 although the merged upstream tag is v0.2.10; why is unclear from evidence); CUDA off by default.
- `F:\REPO_CLONES\Strata-fork\CMakeLists.txt:931-945` - the DSV41 engine is its own opt-in executable and its sources.
- `F:\REPO_CLONES\Strata-fork\CMakeLists.txt:957-971` - build id define, includes, links, native-expert define.
- `F:\REPO_CLONES\Strata-fork\fork\build_windows_cuda.bat:26`, `:31-32`, `:40` - default target, DSV41 switch, flags.
- `F:\REPO_CLONES\Strata-fork\tools\dsv41\pack_gguf.py:3-24` - what a pack is; no weights copied; drafter and vision separate.
- `F:\REPO_CLONES\Strata-fork\tools\dsv41\pack_gguf.py:431` and `:929-945` - the `DSE1` header and the files written.
- `F:\REPO_CLONES\Strata-fork\fork\HANDOFF.md:118-121` and `:128-129` - merge status, offline build, staging.
- `F:\REPO_CLONES\Strata-fork\third_party\project-maya\NOTICE.md:3-5` - provenance of `src\dsv41`.
- `F:\REPO_CLONES\Strata-fork\docs\DEEPSEEK41.md:82-86` - what the Coder is (192 of 384 experts, own profile).
- `F:\REPO_CLONES\Strata-fork\data\dsv41\coder-set.json:4` and `:8-10` - source repo and expert counts.

### Installed artifacts and logs on this PC

- `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:3-18` - engine arguments.
- `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:19-31` - cwd, model name, arch, log, port, ready timeout, host.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:3` and `:10-11` - 40 layers; 192 routed experts, 6 per token.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:33-43` - compress ratios, Engram layers, drafter keys all zero.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:54-55` - `format gguf`, `model_dir`.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\tensors.txt:1-8` - shard table and first tensors.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\tensors.txt:24250-24251` - Engram tables, IQ4_NL, 55.3 GB each.
- `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:1-23` - the engine's own stderr log of the run (lines cited individually below).
- `C:\Users\Ryan\.litesuite\llm\strata\engine-0.2.10\BUILD.json:3`, `:5-6`, `:14-17`, `:48` - branch, build provenance and "not started with a model".
- `C:\Users\Ryan\AppData\Local\Temp\claude\577fb47b719be520\d93f4cd7-bdcc-4373-bff9-ce899da0bc39\scratchpad\t0825\strata-serve.log:1-5` - the server's start-up narration; `:7` the thinking-cap line; `:9`, `:23`, `:34`, `:45`, `:57`, `:118` the per-request `done:` lines.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\engram.bin` - binary; only its first 28 bytes were read (magic and six int32 header fields). Size 517,980 bytes from a directory listing.
- `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\expert-profile.bin` - not opened; size 30,720 bytes and modified time (20:15, the pack's other files 20:00) from a directory listing.

### NInfer (for the comparisons in section 8 only)

- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:204-222` - NInfer expert residency.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:226-246` - NInfer CPU share.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:250-269` - NInfer prompt spans.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:273-285` - NInfer Engram handling.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:526-547` - NInfer's own measured comparison against Strata (other host).
- `G:\REPO_CLONES\ninfer-all\README.md:59` - hybrid prefix cache is the server default.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:565-573` - repeatability and memory notes from that comparison.

## Evidence glossary

- **Pack**: a folder of four small files (`ds4.txt`, `tensors.txt`, `engram.bin`, `tokenizer\`) that indexes GGUF shards left where they are (`F:\REPO_CLONES\Strata-fork\tools\dsv41\pack_gguf.py:3-16`).
- **CPU lane**: the routed experts not held in VRAM, computed by CPU threads from RAM copies while the GPU does its own work (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9573-9598`).
- **VRAM tier / `n_vram`**: the per-layer count of routed experts placed on the GPU (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5446-5475`).
- **Heat**: a per-expert float used to rank experts for swapping; seeded from the profile, decayed every 256 tokens (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5583-5586`, `:8822-8832`).
- **Profile** (`expert-profile.bin`): headerless `u32` routing counts, one row of `n_exp` per layer (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:53-63`).
- **Engram**: DeepSeek's n-gram memory; two tables on layers 1 and 14 hashed from token ids (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:36`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7819-7830`).
- **Window / chunk**: a decode or short-prompt forward of up to 16 tokens vs the layer-major chunked prompt pass from 96 tokens (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_pick_rule.hpp:18`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:12453-12457`; the 16 is `kWin` at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cuh:18`).
- **Mailbox**: the host-mapped request ring (`reqs[...]` with a sequence number) between the GPU's router kernel and the CPU-lane thread (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9387-9393`).
- **Slot**: a kept conversation state (compressed KV included) held in RAM and a folder so a later request that continues it reads only its new tokens (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:3-23`).
- **DSpark**: the speculative drafter. This pack carries none: its drafter keys are 0 (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:38-43`).
- **GSQ-RCO**: appears only as part of the source repo name (`F:\REPO_CLONES\Strata-fork\data\dsv41\coder-set.json:4`); what the letters denote is unclear from evidence.

---

## 1. Framing: what was run, and which figures are on disk

**What it is.** Strata is a private fork (branch `fork/merge-0.2.10`, upstream v0.2.10 merged, engines built offline and staged, "UNVERIFIED by Ryan") (`F:\REPO_CLONES\Strata-fork\fork\HANDOFF.md:118-121`, `:128-129`). Its DeepSeek-V4.1 path is a separate executable, `strata-dsv41`, ported from Project Maya's `maya-ds4` (MIT) and shares no symbols with the Qwen engine (`F:\REPO_CLONES\Strata-fork\CMakeLists.txt:931-945`, `F:\REPO_CLONES\Strata-fork\third_party\project-maya\NOTICE.md:3-5`). It is opt-in at configure time; CUDA is off by default (`F:\REPO_CLONES\Strata-fork\CMakeLists.txt:80`), and the Windows build script turns the DSV41 switch on only when that target is named (`F:\REPO_CLONES\Strata-fork\fork\build_windows_cuda.bat:26`, `:31-32`).

**What the model is.** The Coder keeps 192 of the base model's 384 routed experts in every layer and has its own routing profile (`F:\REPO_CLONES\Strata-fork\docs\DEEPSEEK41.md:82-86`). The installed pack confirms 40 layers, 192 routed experts and 6 per token (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:3`, `:10-11`) in three quant types: counted from `tensors.txt` `t` lines, IQ1_M 11,520 tensors, IQ2_XXS 8,064 and IQ2_XS 3,456 for the routed experts, 66.04 GB in total; the two Engram tables are IQ4_NL at 55.3 GB each (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\tensors.txt:24250-24251`). The totals are from a scan of that file with a one-line awk, not quoted from a document.

**What was launched.** One run config launches `strata-dsv41.exe` with `--max-context 131072 --eos 1 --vision on --dspark off --dspark-conf 0.65 --reserve-mib 900` (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:3-18`). It sets no `--mmap-experts`, `--ssd-cache`, `--prefill-chunk`, `--experts` or `--parallel`, so every one of those uses the engine's automatic rule (defaults in `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:26-77`). Nothing in the config is a tuning flag; the behavior in this study is the engine's defaults.

**The figures, checked against the log.**

| Figure from the request | What is on disk | Source |
|---|---|---|
| 46.8 tok/s decode | Not found as a number. Per-request decode in the engine log runs 33.95 to 51.28 tok/s (excluding the 8-token warm-up at 22.90). The server's own `done:` lines read 22.9, 39.4, 46.0, 47.6, 35.1 and 37.3 tok/s; none is 46.8. | `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:13-23`; `C:\Users\Ryan\AppData\Local\Temp\claude\577fb47b719be520\d93f4cd7-bdcc-4373-bff9-ce899da0bc39\scratchpad\t0825\strata-serve.log:9`, `:23`, `:34`, `:45`, `:57`, `:118` |
| 41 s load | Not found. The log has the start line and the first request, with no ready time; the server printed "still starting (20 s)" so the start took longer than 20 s. The RAM copy alone took 19.5 s. | `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:7`; `C:\Users\Ryan\AppData\Local\Temp\claude\577fb47b719be520\d93f4cd7-bdcc-4373-bff9-ce899da0bc39\scratchpad\t0825\strata-serve.log:1-5` |
| 31.8 GB VRAM | Weights 5.70 GB + experts 23.55 GB + caches 0.12 GB = 29.37 GB, with 431 MiB free of a card the engine calls "32 GiB". Scratch is not itemized there. | `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:3` and `:9` |
| 77 GB RAM | Not found. The log shows 42.8 GB of experts copied into RAM and 74.9 GB free at the time of the decision. A resident total is unclear from evidence (no live memory sample was taken). | `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:5` and `:7` |
| "Straight from the GGUF" | Confirmed: `format gguf`, `model_dir` points at the shard folder on G:, and the engine maps the shards in place. | `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:54-55`; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:445-454` |

Which prompt and settings produced 46.8 and 41 is unclear from evidence; treat those two as Ryan's reported numbers, not as something this study reproduced.

## 2. Core components and the process boundary

Three layers, two process boundaries.

1. **Python server** (`serve\server.py`). Owns HTTP, chat templating, thinking budgets, queueing and the lifetime of the engine child. It never links the engine.
2. **Engine child** `strata-dsv41 --serve`. Owns all device and memory work. It is spawned with piped stdin and stdout and stderr to the log file (`F:\REPO_CLONES\Strata-fork\serve\server.py:1145-1152`).
3. **A pack folder** plus unmodified GGUF shards, which are the only model input (section 4).

**The wire protocol is plain text on stdin/stdout.** At start the engine prints `INFO k=v` lines and then `READY <context> stop` (or `... multi=<n>` with `--parallel`) (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:513-520`). The server folds every `INFO` line into a dict and takes the context size from `READY` (`F:\REPO_CLONES\Strata-fork\serve\server.py:1176-1184`). A request is `GEN <max_new> key=value ... <ids>` (or `GENI` with a pictures file); the engine answers `T <id>` per token and one `DONE ...` line (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:3-23`, `:1433`, `:1475`). Anything else gets `ERR unsupported command` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1362-1366`). Sampling is passed as keys on the GEN line (temperature, top_p, top_k clamped to 1..64, min_p, penalties) (`F:\REPO_CLONES\Strata-fork\serve\server.py:1545-1560`), assembled at `F:\REPO_CLONES\Strata-fork\serve\server.py:2094` and `:2512-2513`.

**Why the split matters for porting.** The engine is a stdin/stdout program with no sockets of its own; the fork's build record says the exes import no network DLL (`F:\REPO_CLONES\Strata-fork\fork\HANDOFF.md:128`). Everything the model-specific code needs from the outside arrives as the pack, environment variables and the GEN line.

**Process lifetime.** Before a start, leftover engines from a lost server are ended so the new one does not load beside an old one still holding tens of GB (`F:\REPO_CLONES\Strata-fork\serve\server.py:133-140`). Shutdown is a ladder: `STOP` and `QUIT`, a wait, terminate, then kill (`F:\REPO_CLONES\Strata-fork\serve\server.py:2821-2840`).

**Conversation reuse and slots.** A request whose prompt continues the engine's current state reuses it; one that shares only the start goes back to the shared point; anything else starts over. With `STRATA_DS4_SLOT_KEEP=1` displaced conversations are kept (state plus compressed KV) and survive across runs (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:3-23`). Defaults are 4 slots in RAM, 16 GB folder cap and 1024-token minimum in the engine (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:549-567`); the server sets keep=1 and a 2 GB folder cap for `deepseek41` (`F:\REPO_CLONES\Strata-fork\serve\server.py:3945-3959`, applied at `:4053-4057`). The log shows both halves working: "9942 reused + 17 read in 252 ms" and "saved a 10091-token conversation" (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:21-22`).

**Multi-conversation mode** is not on in this run (no `--parallel`). When it is, a new prompt goes to the idle conversation that reuses the most of it, and two or more active requests step together without drafts (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1495-1503`).

**Thinking budget (server side).** DeepSeek requests get a default budget of 32768 thinking tokens unless the request or config says otherwise (`F:\REPO_CLONES\Strata-fork\serve\server.py:217`, applied at `:4826`). A budget larger than `max_tokens` minus an answer reserve is cut down to `max_tokens - reserve`, where the reserve is `min(max(512, max_tokens // 4), max_tokens // 2)`, so a small `max_tokens` ends up capped at half (`F:\REPO_CLONES\Strata-fork\serve\server.py:5898-5905`; a separate no-budget rule for 48 to 1023 tokens is at `:218`, `:5890-5895` and did not fire here because the default budget was set). The serve log shows the cut applied ("thinking capped at 8 of max_tokens 16", "200 of max_tokens 400", each "budget 32768") (`C:\Users\Ryan\AppData\Local\Temp\claude\577fb47b719be520\d93f4cd7-bdcc-4373-bff9-ce899da0bc39\scratchpad\t0825\strata-serve.log:7`).

## 3. Lifecycle and data flow

### 3.1 Load (one pass, about the engine's own order)

1. **Parse the pack.** `Pack::load` reads `ds4.txt` into a key/value map, takes `model_dir`, then parses `tensors.txt`: `file <i> <name>` rows, then `t <name> <dtype> <file> <offset> <bytes> <dims...>` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:385-400`, `:401-434`).
2. **Open the shards.** Each file is opened read-only; for `format gguf` each is also mapped whole and given a direct (unbuffered) handle and a parallel-read handle (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:436-454`).
3. **Read the Engram hash tables** from `engram.bin` into RAM (magic `DSE1`, six int32 header fields: vocab, layers, n-gram, heads, head dim, pad). The installed file decodes to `DSE1, 129280, 2, 4, 8, 256, 2` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:456-481`; the file is binary, header decoded from its first 28 bytes at `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\engram.bin`).
4. **Gate the geometry.** Head dim must be 512, index head dim 128, experts per token at most 8, hyper-connection width at most 8; no CUDA device is a hard error (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2654-2672`).
5. **Narrow tensor parallelism if the quant blocks demand it.** The log shows "tensor parallel 2 -> 1" because the experts' 256-value blocks split n_ff 2304 no other way (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2705-2711`; `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:2`).
6. **Decide residency and chunking from bytes, not timing.** A GGUF pack marks its experts as living in the files (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2754`); whether they get copied to RAM is decided in section 4. The prompt chunk comes from a rule function (section 6) and printed "prompts in chunks of 4096 tokens" (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2777-2795`; log line 3).
7. **Load the vision tower before the layer split.** If it fails, pictures are turned off with a warning and the load continues (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2814`). A pack with no tower reads text only (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7694-7697`).
8. **Skip the drafter.** `--dspark off` sets `spec=false` in the main loop but leaves `o.dspark` true; only `none` also skips loading (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2536-2540`). `n_mtp` is zeroed when `o.dspark` is false (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2849`). This pack has `dspark_block_size 0` anyway, so there is nothing to draft with (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:38-43`).
9. **Check dense weights fit.** Attention, shared experts and head always live in VRAM; if they do not fit the free memory the load fails with a message, and if they fit but leave less than the reserve it only warns (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3058-3103`).
10. **Allocate and upload.** Scratch goes through `dalloc` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3194-3216`); compressed-KV and index caches are sized per KV source from the context (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3252-3254`); dense weights upload with one host thread per device (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3304-3310`); the Q5_K embedding table is decoded row by row to BF16 into pinned host memory (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3339-3353`).
11. **Start the Engram reader, read the profile, place experts, start threads.** Reader threads start at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3446`; the profile is read with a size check (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3490-3500`); `place_experts` is defined at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5372`; the CPU-lane thread is pinned to a core and the cache-swap threads started (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3757-3760`, `:3778`).
12. **Print the readiness line and serve.** `main` loads then calls `serve()` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2621-2629`).

`--pack-info` is the one path that does none of this: it reads only the two text files and needs no device (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2602-2608`).

### 3.2 One request

- The server writes a GEN line; the engine reuses the shared prefix, reads the rest of the prompt, samples the first token, then loops: `T <id>`, EOS check, next round (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1355-1368`, `:1415-1436`).
- With no drafter, each round is `m.step(tok)` followed by `m.sample(...)` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:330-333`).
- After the request, the engine prints a stats line (reused vs read tokens, tok/s, "CPU lane x% of the experts") and calls `save_profile()` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1472-1493`).

### 3.3 One decode token through the model

- `step` is `forward(&token, 1, ...)` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:11375-11380`). `forward` checks the context limit and token range, then **issues the Engram row reads for the whole window first** so they overlap with the embedding and earlier layers (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13359-13410`).
- `plan_window` selects the async MoE path when decode is not forced synchronous and the window is at most `async_max_t` = 8 tokens (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13230-13234`, `:1316-1323`).
- `forward_serial` embeds, then walks layers; on the single GPU all 40 are on CUDA0 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13441-13449`; log line 9 shows `layers [0, 39]`).
- **Per layer:** on the two Engram layers the rows are collected just before they are needed (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13999-14001`), then attention, then the MoE sequence below.
- **MoE, async path** (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14122-14146`):
  1. gate matvec gives the 192 logits;
  2. with lookahead on, the **next layers'** gates are also run on this layer's FFN input and published as guesses (`:14130-14139`);
  3. `route_publish` picks the top-k, writes each pick's VRAM pointer to the plan (null if the expert is not on the card), and publishes the picks to the host mailbox (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1607-1625`, `:1655-1673`);
  4. meanwhile the shared expert runs on the GPU (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14144-14146`).
- **Expert delivery** (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14176-14196`): `moe_gate_up` runs for experts with a non-null pointer; the GPU then waits on the CPU lane's sequence number (`wait_seq`); `moe_down` sums VRAM experts, the CPU lane's partial and the shared expert.
- **CPU side, in parallel:** `serve_lane` spins on the request ring, runs `cpu_experts` on the RAM copies, sums `weight * y` into the response partial and releases the response with a fence and the sequence number (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9387-9393`, `:9573-9598`).
- `hc_post` writes the residual, and after layer 39 the head matvec and `sample` run on the head device (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14248`, `:14453-14456`).

**The PCIe streaming lane is not part of this run.** It is opt-in by `DS4_STREAM`; the log says "off (v1.1.27's path: DS4_STREAM=0, nothing streams)" and `lane_alloc` returns early (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9716-9720`, `:9753-9754`; log line 4). The server config sets no `env` (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:3-31`).

### 3.4 The router kernel (the mechanism everything else hangs on)

Score is `sqrt(softplus(logit))`, the pick is the top-k of `score + bias` (ties go to the lowest index, and a non-finite gate input takes a fallback pick), and the weight is `score / sum(score) * route_scale`. For each pick the kernel writes `plan_ptr = vram[pick]` (zero means CPU lane), records the pick in the host request, and copies the token's FFN input to the host only if some pick is on the CPU lane, because that copy is 20 KB per row over PCIe (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1621-1625`, `:1629-1651`, `:1655-1673`).

## 4. Memory placement: where each byte lives

### 4.1 The pack as an index

The pack is metadata only. `pack_gguf.py` writes `ds4.txt`, `tensors.txt`, `engram.bin` and the tokenizer folder "beside the model's files (no weights copied)", and says a GGUF has no drafter or vision tower of its own (`F:\REPO_CLONES\Strata-fork\tools\dsv41\pack_gguf.py:3-24`; files written at `:929-945`; header packed at `:431`). The installed Coder pack points five Coder shards (files 0-4) plus an Engram GGUF and an mmproj at `..\` paths, resolved against `model_dir` on G: (`C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\tensors.txt:1-7`; `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\ds4.txt:55`; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:436-454`). Shards are never copied or rewritten.

### 4.2 The four placements in this run

| Data | Placement | Evidence |
|---|---|---|
| Dense weights (attention, shared expert, head) | VRAM, 5.70 GB, must fit or the load fails | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3058-3103`; log line 9 |
| Routed experts, hottest | VRAM, 23.55 GB, 2,701 of 7,680 experts | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5446-5475`; log line 9 |
| Routed experts, the rest (4,979) | Anonymous RAM copy, 42.8 GB, 1 NUMA part, copied in 19.5 s, computed by 16 CPU threads | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5834-5883`; log lines 7 and 10 |
| Engram tables (110.6 GB) | Stay in the shard file; rows are read per token and cached | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7819-7899` |
| Embedding table (Q5_K) | Decoded to BF16 in pinned host memory (about 1.32 GB by arithmetic: 129280 x 5120 x 2 B, not logged) | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3339-3353` |
| Vision tower | Host RAM; borrows about 320 MiB of VRAM (header comment) from the `--reserve-mib` room only during a picture turn | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:70-72`; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2620`; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7733` |
| KV (compressed) | VRAM, 0.12 GB at 131072 context | `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3252-3254`; log line 9 |

### 4.3 How the VRAM share is computed

`place_experts` takes `(free VRAM - reserve) x vram_experts`, gives each layer `floor(frac x n_exp)` experts, then hands out the remaining whole experts layer by layer (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5446-5475`). The profile's counts seed each layer's heat (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5583-5586`). On this PC `--reserve-mib 900` was in the config because vision is on (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:16-17`).

### 4.4 The all-or-none RAM copy

The rule is a single boolean function: the CPU lane's bytes plus an 8 GiB margin must fit available RAM, and (on Windows) the commit room (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:99-106`; margin at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1234`). `store_gb` calls `cpu_lane_fits_ram` and, if true, sets `auto_copy_all` and turns the demand store off, because "the demand store would only be a second, smaller copy" (the store is a RAM tier of whole experts read by unbuffered requests into slots: `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_store.hpp:1-9`) (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:6469-6485`, `:6486-6508`). `copy_experts` then copies all or none: `budget = auto_copy_all && all <= can ? all : 0.0` (the branch taken when `--mmap-experts` is not given; `--mmap-experts off` takes `min(all, can)`) (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5872`). The log shows the decision: "43.2 GB fit the RAM ... 43.6 GB beside 9 GB for the rest: copied in at load, no demand store" (log line 6).

A reading aid: the comment above `copy_experts` says auto means "none", while the code at line 5872 and the message at 5881-5883 do the all-or-none copy when it fits (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5834-5840`). The code and the log agree with each other, so this study follows the code.

Whether the copies were pinned is unclear from evidence. The rule exists (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:190-196`: pin only when the arena is on explicit huge pages, or RDNA4 with at least half the experts on cards), and the log has neither the "pinned" nor "pinning failed" line, so it is probably not pinned, which is an inference.

### 4.5 The Engram path

Rows are fetched by token id and a per-head hash. `eg_issue` looks each row up in a clock-evicting cache, de-duplicates repeats within the window, and queues a file read of `id x row_bytes` for the rest (`hd/32*18` = 144 bytes for IQ4_NL) (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7819-7899`, in particular `:7860-7871`, `:7881-7883`). The reader runs on its own threads with their own file handles because one synchronous Windows handle serializes reads (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_engram_io.hpp:3-10`). The cache is sized as RAM/64 MB clamped to 16..256 MB (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3417-3420`); direct I/O is off by default and turned on by `DS4_EG_DIRECT` (`:3426-3429`). The rows are requested when the window starts and collected right before layer 1 and layer 14, so the reads hide behind earlier layers (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13359-13410`, `:13999-14001`).

### 4.6 What "straight from the GGUF" costs at load

Because experts are never converted, the cost is the 19.5 s RAM copy plus the dense upload and embedding decode; the log shows the copy and not the rest as a separate time (log line 7). Per-layer expert blobs must have `w1`, `w3`, `w2` at fixed strides inside one file, which is what the GGUF's one-tensor-per-role layout supplies (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2386-2392`).

## 5. Performance techniques (the mechanisms behind the decode rate)

1. **Overlap GPU and CPU on the same layer.** The router publishes picks to the host mailbox, the GPU runs the shared expert and its resident experts, the CPU lane runs the rest, and the GPU waits only before the down projection (section 3.3; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14122-14196`, `:9573-9598`).
2. **Adaptive VRAM cache.** Each layer keeps a heat value per expert. Each tick, the hottest CPU-lane expert that was hit is swapped in for the coldest VRAM expert, but only if `heat[best] > heat[cold] * 1.25 + 1.0` (defaults), at most 8 swaps in flight (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9254-9300`, tunables at `:1349-1355`). Heat decays by 0.9 every 256 tokens (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:8822-8832`). Swaps need a spare VRAM slot in the layer (`spare_vram >= 0`, `:9266`) and run on dedicated threads (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3778`).
3. **Lookahead routing.** The device also runs the next one or two layers' gates on the current FFN input and publishes the guesses with the current routing, so the CPU lane can prepare (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14130-14139`). The CPU lane also orders its experts by a per-expert flag from `ahead_fl`, which is named for the lookahead (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9561-9569`); what flag bit 8 encodes is unclear from evidence.
4. **Engram I/O off the critical path** (section 4.5).
5. **Chunked prompt pass.** From 96 tokens a prompt goes through the layer-major chunked pass; below that, windows of 16 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_pick_rule.hpp:18`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:12453-12457`). In the log this gave 9,792 tokens in 8,896 ms (1,100.7 tok/s) (log line 20). During the chunked pass the experts stream to the GPU a chunk at a time instead of being computed on the host (comment at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:64-67`).
6. **Prefix and slot reuse** so an agent loop re-reads only its new tokens (section 2; log lines 15-22).
7. **Persisted routing profile** so the next run places experts by what this workload actually routes (section 6).

**What the log shows about the CPU lane share.** The per-request "CPU lane x% of the experts" falls from 41.7% on the first short request to 20.1% by the seventh and 3.9% on the 9,792-token prompt (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:13-20`). The swap machinery above is the mechanism that moves experts onto the card, but the log does not attribute the drop to it, and the long-prompt figure is also affected by the chunked pass streaming experts to the GPU, so the log alone does not separate the two causes (unclear from evidence).

## 6. Configuration and policy

### 6.1 Where each decision comes from

- **Engine flags** are parsed in `main`; unknown flags exit 2 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2597-2599`); `--dspark` and `--vision` are parsed at `:2536-2540` and `:2551`. The defaults and their rationale live in `Options` with long comments (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:26-77`).
- **Run config** is a JSON the server reads: `exe`, `args`, `cwd`, `tokenizer`, `model_name`, `arch`, `log`, `lib_dirs`, `port`, `engine_ready_s`, `gpu`, `host` (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:2-31`). `engine_ready_s` of 3600 overrides the 900 s default because a very large model may need more than a quarter hour (`F:\REPO_CLONES\Strata-fork\serve\server.py:378`, `:3417-3425`).
- **Environment** carries the rest: slots (set by the server for this arch), cache tunables, `DS4_STREAM`, copy and store sizes (`F:\REPO_CLONES\Strata-fork\serve\server.py:3945-3959`; `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:1349-1355`).

### 6.2 Rules as pure functions with their measurements inline

`ds4_experts_rule.hpp` holds the sizing decisions as small header-only functions, each with the benchmark that justified it in the comment above:

- `experts_per_word`: with no `--experts` given, 5 experts when the GPU is integrated, an RDNA4 case applies, or total VRAM is at most the 16 GiB faster-mode limit, otherwise the model's 6 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:43-51`).
- `prefill_chunk_default`: 1024 when the cards hold half the experts or more and the run is a tp-1 default or a multi-card RDNA4 tp run (neither holds here: the engine's tp is 2 and the card is NVIDIA), 4096 when a GGUF's experts are RAM-copied on a non-HIP build and the smallest card has at least 10 GiB (the branch this run took), 4096 for in-place experts on cards under 16 GiB, otherwise 2048 or the big chunk (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:65-97`). The log prints 4096 at line 3 and "prompts in chunks of 32768" at line 5; the code applies `chunk_default` after the rule when `--prefill-chunk` is not given (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2788-2795`), and the later line is the one that reports the working size: `chunk_default` can raise the chunk to `kMaxChunk` (32768) when the experts stay in the files and the free VRAM allows, and the startup line that prints the chunk comes after it (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3993-4017`, `:3585-3590`). Why `chunk_default` returned 32768 although `gguf_copy` was true is unclear from evidence: its guard reads `file_experts`, which `ds4_model.cu:2754` sets for every GGUF pack, so the copy decision does not stop it.
- `copy_all_fits` (section 4.4).
- `profile_size_fits`: accepts the file only if its size is between `layers` and `layers + 8` rows of `n_exp` u32 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:53-63`). The comment explains why: a profile counted on 384 experts per layer would otherwise be read silently as rows of 192 and give every layer the wrong experts. The installed profile is 30,720 bytes = 40 x 192 x 4, which passes (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3490-3500`; size from a directory listing of `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\expert-profile.bin`, not a line).

### 6.3 The profile has a runtime effect

`save_profile` writes the live routing counts back to the pack's `expert-profile.bin` through a temp file and an atomic replace unless `--profile-readonly` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14802-14812`); it is called after every request (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1493`). `Options` documents the consequence: the next run places experts differently, so greedy output can move between runs (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:36-39`). On this PC `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\expert-profile.bin` is dated 20:15 and the pack's other files 20:00 (directory listing), which fits that write-back (the engine log's start line is 20:12:31: `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.log:1`); whether the file existed before that run is unclear from evidence.

### 6.4 Build configuration

Static library `strata_dsv41_kernels` (vendored `exl3`, `sampler`, `moe_win`) plus the executable from 13 sources; `DS4_BUILD_ID` is compiled into `ds4_main.cpp`; links the prefill library, CUDA runtime, cuBLAS and Threads; with `STRATA_NATIVE_EXPERTS` also links the CPU trellis kernels (`F:\REPO_CLONES\Strata-fork\CMakeLists.txt:936-945`, `:957-971`). The staged build used CUDA architecture `120a`, nvcc 13.0.88 and MSVC 19.38 (`C:\Users\Ryan\.litesuite\llm\strata\engine-0.2.10\BUILD.json:14-17`). `--pack-info` is the host-only inspection path (section 3.1).

## 7. Error handling and supervision

- **Fail early, with a plain message, on conditions that cannot work.** No CUDA device; geometry outside what the kernels cover; dense weights larger than free VRAM (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2654-2672`, `:3058-3103`). Load failure prints `maya-ds4: <err>` and exits 1 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2621-2629`); the server turns that into `EngineDied` with the exit code and the log tail (`F:\REPO_CLONES\Strata-fork\serve\server.py:1228-1240`).
- **Degrade rather than fail where a slower path exists.** Prompt buffers that do not fit halve the chunk (down to a floor) and fall back to windows of 16 (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3523-3530`); a vision tower that fails to load disables pictures only (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:2814`); a RAM copy that does not fit leaves experts read in place (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5874-5880`); a profile of the wrong size is ignored and experts start unranked (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:3497-3499`).
- **Silent substitution to know about.** A failed Engram row read becomes zeros and is not cached; the warning for it exists only in the Metal build (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7930-7938`). This is a design fact, listed for porters.
- **Numerical sanity check in the serving loop.** Every 8 tokens (and at the first) the engine asks whether the router saw non-finite scores; if so it ends the request with an `ERR`, sends no `DONE`, resets state and drops the conversation (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1415-1427`, `:1489-1492`; the check itself is `router_nonfinite` at `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:8761`). The code comment says it is one small copy from the device, not a per-token sync.
- **Fatal GPU errors end the process on purpose.** On messages such as "illegal memory access" or "device lost" the engine prints `ERR fatal GPU error` and `_Exit(3)`, skipping teardown because freeing memory on a dead context can hang (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:390-396`). The server restarts it on the next request but stops after 3 such deaths in 600 s and returns 503 until the window passes (`F:\REPO_CLONES\Strata-fork\serve\server.py:4953-4979`).
- **Memory-pressure behavior.** The demand store gives RAM back in 0.7 GB chunks when free RAM drops below 4 GB (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:8802-8821`); it is not active in this run because the copy-all rule disabled it (section 4.4).

---

## 8. NInfer design notes (adaptation guidance)

NInfer-all already serves DeepSeek-V4.1-Flash with host or disk experts, a CPU share, prompt spans and an Engram reader (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:204-222`, `:226-246`, `:250-269`, `:273-285`). The notes below say what Strata does that is different or additive and what would have to hold for a port. They do not describe NInfer's source (not read in this pass beyond its own docs).

**Pattern takeaway for the pack / model-input boundary:**

- Strata's whole model input is four small text/binary files that index untouched GGUF shards, resolved through a `model_dir` line and relative `..\` shard names (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:385-400`, `:436-454`; `C:\Users\Ryan\.litesuite\llm\strata\packs\deepseek-v4.1-flash-ista-das-coder\tensors.txt:1-7`). The benefit shown in this deployment: the Engram table, the mmproj and the Coder shards are separate G: files shared across packs, with no conversion step and no copy. NInfer's docs already keep the Engram table as its own artifact (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:277-281`); the part worth copying is the plain-text tensor table with a `--pack-info` path that needs no device (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2602-2608`).
- Keep drafter and vision as optional additions that the pack builder discovers, so a pack without them runs text-only without a code branch per quant (`F:\REPO_CLONES\Strata-fork\tools\dsv41\pack_gguf.py:18-21`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7694-7697`).

**Pattern takeaway for CPU/GPU expert execution:**

- Strata computes every routed expert that is not on the card on CPU threads from RAM copies, concurrently with the GPU's shared expert and resident experts, and the GPU waits on one sequence number before the down projection (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:14122-14196`, `:9573-9598`). NInfer's CPU share is a split for decode or verification calls of up to eight tokens, and `auto` copies every miss of disk experts (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:226-246`). NInfer's own measurement against Strata on a different host found Strata's decode 1.22 to 1.23 times faster after a long prompt with all six experts and no drafts, and notes that Strata copies no missing expert while its CPU threads compute them (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:539-547`, `:562-564`). The porting candidate is the policy "compute all misses on CPU when the full lane fits in RAM"; the host in this study (16 CPU threads, per log line 10) is much smaller than the host in that comparison (61 CPU threads, same file line 516), so which split wins here is unclear from evidence.
- The router kernel's trick of writing the VRAM pointer into the plan (null means CPU) and copying the FFN input to the host only when a pick needs it is a small, portable idea (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_kernels.cu:1655-1673`).
- Constraint to preserve: NInfer states its tokens and log probabilities do not depend on the split or thread count because the sums are fixed-point and order-independent (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:235-242`, `:565-572`). Strata's own path gives no such guarantee and notes run-to-run movement from profile write-back (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.hpp:36-39`). Any CPU-lane port must keep NInfer's integer-sum arithmetic.

**Pattern takeaway for the expert cache policy:**

- Heat with hysteresis (swap only when `hot > cold x 1.25 + 1`), exponential decay on a token count (x0.9 per 256), a cap on in-flight swaps and a per-layer spare slot are all in one place and all tunable by environment (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9254-9300`, `:8822-8832`, `:1349-1355`). NInfer's cache is "what decode routes most", filled at startup from a profile (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:214-219`); the evidence available does not say whether NInfer re-ranks during a run, so whether a hysteresis rule adds anything is unclear from evidence.
- The starting heat is the profile scaled to about 512 hits per layer, so the profile and the live counts share a scale (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:5583-5586`).

**Pattern takeaway for sizing rules:**

- Put every residency and chunk-size decision in small pure functions with the benchmark that justifies each branch in the comment (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:65-97`), and make every rule produce one printed line at load so a log reader can see which branch ran (log lines 3, 5, 6, 8). A new box then gets a rule revision rather than a flag.
- The all-or-none RAM copy is the best-evidenced single choice here: one boolean on bytes plus a margin, no timing, and it removes the second cache (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_experts_rule.hpp:99-106`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:6486-6508`). The 8 GiB margin was enough here (43.6 GB available beside 9 GB for the rest, log line 6) but no live memory sample was taken, so whether the system stayed responsive under it is unclear from evidence.

**Pattern takeaway for the Engram reader:**

- Issue every Engram row read when the window's token ids are known and collect only just before the consuming layer, with a bounded clock-eviction row cache and readers on their own file handles (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_engram_io.hpp:3-10`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:13359-13410`, `:13999-14001`). NInfer also reads rows from the file by default, with an optional row cache (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:282-285`); compare issue time against consume time, not the cache.
- Decide explicitly what a failed row read does. Strata zero-fills silently outside Metal builds (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7930-7938`); NInfer documents that running without the table degrades perplexity from 3.07 to 9.68 (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:279-281`), which is a reason for a port to fail loudly instead.

**Pattern takeaway for conversation reuse:**

- Prefix reuse plus kept slots (RAM first, then a folder with a cap and age limit, text-only) turn an agent loop's repeated prompts into 17-token reads (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:549-567`; log lines 15, 21-22). NInfer has prefix reuse (`G:\REPO_CLONES\ninfer-all\README.md:59` lists the hybrid prefix cache as the server default) but this study did not read how it persists across restarts, so any difference is unclear from evidence.

**Pattern takeaway for the process boundary and supervision:**

- A line-protocol child with `READY` and `INFO` lines, a ready timeout set per model, leftover-engine cleanup before a start, and a shutdown ladder (`F:\REPO_CLONES\Strata-fork\serve\server.py:133-140`, `:1176-1184`, `:3417-3425`, `:2821-2840`) isolates GPU faults from the HTTP server. The deliberate fast exit on a fatal CUDA error and the restart budget (3 in 600 s) are the part to copy if NInfer lacks them (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:390-396`, `F:\REPO_CLONES\Strata-fork\serve\server.py:4953-4979`).
- The cheap NaN guard (poll the router every 8 tokens, end the request with an `ERR` and drop the conversation) is separable from everything else (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:1415-1427`).

**Pattern takeaway for vision:**

- Keep the tower in host RAM and borrow its VRAM from the reserve only for a picture turn, so a text-only workload pays nothing on the card (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:7733`, `F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_main.cpp:2620`). This is why `--reserve-mib 900` exists in the config (`C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:16-17`). Whether NInfer's image path does the same is unclear from evidence.

**Pattern takeaway for what not to copy blindly:**

- Several Strata features are off in this run and are not evidence for the 46.8 tok/s: the PCIe streaming lane, the demand store, drafting and `--parallel` (`F:\REPO_CLONES\Strata-fork\src\dsv41\ds4_model.cu:9716-9720`, `:6486-6508`; config at `C:\Users\Ryan\.litesuite\llm\strata\strata-ista-das-coder.json:12-13`). A port benchmarked against this deployment should compare only the active set: the VRAM tier, the RAM-resident CPU lane, the heat swap, lookahead routing, the chunked prompt pass, the Engram reader and slot reuse.
- Strata's comments and code can disagree after a policy change (section 4.4). Port from the behavior the log confirms, not from the comment.
