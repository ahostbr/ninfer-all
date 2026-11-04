# Case Study: NInfer Patterns for the Strata port

## Intent

- Describe how NInfer (the `ninfer-all` fork, branch `fork/audited`, HEAD `02c91fce`) would run the pruned DeepSeek-V4.1-Flash "Coder" on Ryan's PC: RTX 5090 32 GB, 96 GB RAM, Windows 11, WDDM.
- Serve as a pattern reference: for each stage of a MoE-with-offloaded-experts engine, show where it sits in NInfer so a Strata port knows where its own tricks plug in.
- Record what NInfer already took from Strata, so the port does not rebuild it.
- Scope exclusion: this is not a bug hunt and not a review. Behaviors that look like limits are described as behavior.
- Nothing was built, run, loaded or timed. Every number from a doc is marked as a doc claim. Anything not read or not traceable is marked "unclear from evidence".

## Scope and evidence sources

All paths are under `G:\REPO_CLONES\ninfer-all`. Each line below was opened and the line number confirmed.

Build and manifests
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:1`, `:60` - minimum CMake 3.28; `project(ninfer LANGUAGES C CXX CUDA)`.
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:77-83` - fork option `NINFER_ALLOW_CUDA_13_0`; a 120a build needs CUDA 13.1+ unless that option and CUDA 13.0 are both set.
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:89-93` - MSVC below 19.40 is refused (`core::wide_add` miscompile); toolset 14.44 is named.
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:105-115` - a 120a build takes the sm8x `mma.sync` compat path unless `NINFER_SM120_NATIVE` is ON.
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:140-142` - on Windows, TMA descriptors are staged.
- `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:187-194` - `NINFER_DIRECTSTORAGE` is refused in this fork (it fetched from nuget at configure time).

Model load and placement
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:65-109` - `expert_blobs`: experts stay as stored GGUF rows, as pointers into the pinned parent (Host) or file extents (Disk), never repacked.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:158-172` - binder setup, `offload` flag, embedding, head and final norm bound through `Bindings`.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen3_5\load\bindings.h:41-48` - `Bindings::parameter` and `direct` default to `artifact::Residency::Device`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:181-190` - per-layer hyper-connection weights bound with `b.direct`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:258-275` - experts bound `Pinned` (Host) or `Values` (Disk) when offloading, resident otherwise.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:276-285` - Engram projection and gate bound only on Engram blocks.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:388-404` - `materialize`, then one `ExpertBlob` pair per expert, plus the alias table.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:416-421` - in Disk mode the expert file list is the artifact and its part files.
- `G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:209-214` - host bytes are read with direct reads on several threads so the page cache does not hold a second copy.
- `G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:808-816` - the host block is a `HostPages` of the planned capacity.
- `G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:850-890` - bytes are read first, then the block is pinned.
- `G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:39-56` - `HostPages` refuses a size above available physical memory, then `VirtualAlloc`.
- `G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:68-99` - `pin()` touches pages on up to 16 threads, then registers them with CUDA.
- `G:\REPO_CLONES\ninfer-all\src\core\host_memory.cpp:148-154` - Windows available memory is `GlobalMemoryStatusEx.ullAvailPhys`.
- `G:\REPO_CLONES\ninfer-all\src\core\arena.h:45-50` - the WDDM evictable budget commits every owning arena at creation.
- `G:\REPO_CLONES\ninfer-all\src\core\arena.cu:401-443` - `PinnedHostBuffer`: `cudaHostAlloc`, with an error text that says a Windows pin competes with VRAM.

Engine core and decode
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\engine.cpp:44-49` - DeepSeek-V4.1 is one of four "own family" cores.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\engine.cpp:459-471` - DeepSeek branch of `Engine::Impl`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:62-72` - `request_error`: which exceptions count as per-request.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:76-90` - artifact detection; an Engram-table artifact passed as the model is rejected.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:118-144` - only `--spec dspark` or `--lookup-ngram`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:145-175` - option validation: BF16 KV, no Qwen-only hybrid options, residency rules, one device unless tensor-parallel.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:176-215` - Engram option rules and fail-fast table lookup before load.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:217-225` - `load_model` receives the residency.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:246-253` - device sync, then free VRAM measured after weights.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:271-293` - `ExecutorOptions` filled from `EngineOptions`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:308-317` - routing profile: explicit file, else built-in by family and widths.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:321-340` - prefill chunk 2048/1024/512 fallback; then `warm_up`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:397-418` - startup diagnostics on expert placement and cache seeding.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:712-728` - pinned host-KV tier created when prefix reuse is on.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:956-972` - `fail_engine`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:974-1000` - worker `loop`: wait, `admit`, `step`, `check_progress`, and `catch (...)` into `fail_engine`.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:1499-1502` - after sampling and a stream sync, `check_experts()` is called.
- `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:1032-1043` - `fail` rethrows anything that is not a request error.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:400-404` - `growing`: elastic expert cache plus growing KV, needs VMM granularity.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:455-484` - up-front KV sizing check with an explanatory error.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:600-621` - Engram scratch, workspace check, workspace allocation, `plan_delivery`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:831-883` - `plan_delivery`: delivery options, CPU-share decision, `ExpertDelivery` construction.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:987-992` - `kGraphReserve` 320 MiB, `kKvHeadroom` 256 MiB, `start_reserve`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1004-1024` - `plan_span`: prompt-span sizing against free memory (the loop at `:1015-1022` tries 8, 4, then 2 chunks).
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2481-2493` - `forward_span`: when `borrow_span` returns nothing, the span runs as two halves of whole chunks.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1083-1130` - KV growth `back`, and `borrow_span`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1441-1469` - `read_engram`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1524-1539` - `engram_block`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1811-1830` - `ffn`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1996-2005` - `routed`: router GEMM then route kernel.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2058-2075` - sort routes, `delivery->begin`, expert tables.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2096-2140` - vector path (cached, shared, await, missing, CPU add) and wide path.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2326-2358` - `run(Pass)` layer loop.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2599-2644` - `decode` and `run_graph`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2793-2855` - `warm_up`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2986-2990` - `check_experts`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\kernels.cu:1031-1032` - route kernel limits (512 experts, top-k 8); `:1508-1509` throws when exceeded.

Expert transport and tier
- `G:\REPO_CLONES\ninfer-all\include\ninfer\model_enums.h:61-71` - `ExpertResidency`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:48-56` - `ExpertCpuProducts` interface.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:84-107` - `ExpertDeliveryOptions`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:176-201` - the design comment: pools, narrow vs wide calls, banks, lending.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:565-576` - disk-mode staging buffer, file streams, `FileReadQueue`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:606-614` - bank must fit in `free - keep_free_bytes`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:669-679` - elastic cache via `GrowableRegion`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1050-1055` - `add_piece` coalesces copies.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1136-1150` - `flush_pieces`: pinned pieces by async H2D, others via staged file reads.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1530-1535` - `serve` on the host service thread.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1562-1587` - wide/narrow bookkeeping, bank lending, CPU share selection.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1758-1778` - `fail` keeps the first exception and neutralizes the call.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1780-1813` - service-thread loop.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1955-1962` - `begin`; `:1994-1999` launches the `split_routes` kernel on the compute stream.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:131-239` - `split_routes` kernel: counts routed experts, calls a call wide at half the layer's experts or more, marks each as hit or miss against the cache address range, and at `:237` publishes the request to the host service thread.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:2109-2115` - `prime`.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:1-21` - tier policy and the Strata attribution.
- `G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:31-37` - tier constants.
- `G:\REPO_CLONES\ninfer-all\src\core\file_read_queue.cpp:95-112` - Windows IOCP overlapped file reads.
- `G:\REPO_CLONES\ninfer-all\src\core\file_read_queue.cpp:191-204` - Windows `ReadFile` enqueue.
- `G:\REPO_CLONES\ninfer-all\src\models\expert_profile.cpp:60-78` - profile writer and `builtin_expert_profile` matching.

Engram table
- `G:\REPO_CLONES\ninfer-all\include\ninfer\engine_options.h:423-440` - `NgramTableOptions`.
- `G:\REPO_CLONES\ninfer-all\include\ninfer\engine_options.h:487` - default residency `Device`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.h:22` - row size 144 bytes.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:73-100` - `stored_rows`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:113-140` - `engram_table_layout`.
- `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:190-202` - the row hash.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.h:52-70` - `NgramTableReader` API.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:100-112` - Windows open flags.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:368-376` - `VirtualLock` path.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:642-648` - residency selects RAM, hot, or a row cache.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:944-948` - `read_rows` is submit then wait.
- `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.h:91`, `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:950` - the reader's `prefetch(row_ids)` hint.

API surface and serving
- `G:\REPO_CLONES\ninfer-all\apps\serve\main.cpp:295-306` - bind first, 503 while loading.
- `G:\REPO_CLONES\ninfer-all\apps\serve\main.cpp:425-440` - exit codes 0, 1, 2.
- `G:\REPO_CLONES\ninfer-all\src\serve\http_server.cpp:469-484` - readiness gate returns 503 `model_loading`.
- `G:\REPO_CLONES\ninfer-all\src\serve\http_server.cpp:1547-1567` - engine watch.
- `G:\REPO_CLONES\ninfer-all\src\serve\serve_options.cpp:1137-1160` - `--expert-residency`, `--expert-cache-mib`.
- `G:\REPO_CLONES\ninfer-all\src\serve\serve_options.cpp:1464-1466` - `--wddm-evictable-budget`.
- `G:\REPO_CLONES\ninfer-all\src\core\device_launch.cpp:15-20` - `cuda_check` aborts.

Docs and fork notes
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:1-12` - what the Coder is; nothing loaded or served yet.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:20-29` - Coder has 192 experts per layer.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:48-60` - router is already cut; built-in profile finds no match.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:115-118` - sizes.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:136-146` - Windows working-set and zero-fill findings.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:156-176` - RAM budget and first-run flags.
- `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:180-189` - vision and drafter components.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:12-30` - model summary.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:190-222` - memory layout and expert residency.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:224-246` - the CPU share.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:271-285` - Engram table.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:300-306` - the drafter round and `--draft-confidence`.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:487-504` - prefix reuse.
- `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:508-566` - NInfer vs Strata on one rented host.
- `G:\REPO_CLONES\ninfer-all\CONTRIBUTORS.md:72-88` - what NInfer took from Strata.

## Evidence glossary

- **Coder**: the ISTA-DASLab GSQ-RCO pruned DeepSeek-V4.1-Flash, 192 of 384 routed experts per layer, IQ2_XXS, with a separate 110.6 GB Engram table (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:3-8`).
- **Artifact (`.ninfer`)**: NInfer's own container, read by `artifact::Reader`; a table of components, tensor objects and bindings; multi-file via part files (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:76-90`, `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:416-421`).
- **Residency**: where routed experts live: `Device`, `Host` (pinned RAM) or `Disk` (artifact files read through the OS page cache) (`G:\REPO_CLONES\ninfer-all\include\ninfer\model_enums.h:61-71`).
- **Pinned**: host memory registered with CUDA so it can be copied asynchronously (`G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:68-99`).
- **ExpertBlob**: one expert's bytes, as a pinned pointer or file extents (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:91-107`).
- **Layer bank**: a whole-layer device window for experts the cache does not hold (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:91-94`).
- **Cache slot / pool**: a device slot in the persistent expert cache; slots form pools by blob size (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:176-179`).
- **Narrow call / wide call**: a decode or verification call vs. a call that routes to half the layer's experts or more, such as a prompt chunk (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:181-195`).
- **Tier**: the policy that decides which experts the cache holds, by counted routes and periodic swap rounds (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:1-18`).
- **CPU share**: part of a narrow call's missing experts computed on CPU threads instead of copied (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:224-229`).
- **Engram**: per-token n-gram rows (IQ4_NL, 144 bytes each) fetched from a large table before each pass and gated into the residual stream on two blocks (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:24-25`, `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.h:22`).
- **Prompt span**: several prompt chunks run as one unit, sized against free VRAM (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1004-1024`).
- **WDDM**: the Windows display driver model; per the code's own comment, a pinned host allocation is mapped into the GPU address space and competes with the card (`G:\REPO_CLONES\ninfer-all\src\core\arena.cu:416-424`).

---

## 1. Framing: what runs, and how it would be built and launched here

NInfer is a CMake project with C, C++ and CUDA languages (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:60`; minimum CMake 3.28 at `G:\REPO_CLONES\ninfer-all\CMakeLists.txt:1`). What is confirmed for a 5090 build:

- The only Blackwell target is `120a`, and it needs CUDA 13.1+; the fork lets CUDA 13.0 through behind `NINFER_ALLOW_CUDA_13_0` (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:77-83`).
- A `120a` build uses the sm8x `mma.sync` compat path by default; the native sm120 routes are an opt-in (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:105-115`). So on the 5090 the default NInfer kernels are the Ampere-style ones, not upstream's Blackwell-native ones.
- MSVC below 19.40 is refused because of a `core::wide_add` miscompile; the message names toolset 14.44 (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:89-93`).
- On Windows, TMA descriptors are staged (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:140-142`) and DirectStorage is cut from the fork (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:187-194`).

How the Coder maps onto the engine:

- The Coder is a 192-expert-per-layer model (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:20-29`). The router is already cut to the kept experts in the GGUF, so the engine sees an ordinary 192-expert MoE (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:48-54`). The route kernel accepts up to 512 experts and top-k up to 8 (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\kernels.cu:1031-1032`), and throws "at most 8 of 512 experts" beyond that (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\kernels.cu:1508-1509`).
- The routed experts are 61.50 GiB, everything else about 5.66 GiB (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:115-118`). A 32 GB card cannot hold 61.5 GiB, so the Coder needs `--expert-residency host|disk`; `device` is the default (`G:\REPO_CLONES\ninfer-all\include\ninfer\engine_options.h:487`) and would be a mismatch.
- Status: the Coder has not been loaded or served by anyone on this fork. The conversion was in progress and "Nothing has been loaded or served" (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:10-12`). Everything below about the Coder on this PC is the code path, not a measurement.

Pattern takeaway: NInfer treats "model family" as a core of its own (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\engine.cpp:44-49`), chosen by artifact sniffing (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\engine.cpp:459-471`). A Strata engine would sit as one more such core, or as the transport under one.

## 2. Core components and layering

The DeepSeek path crosses four layers. Each one owns one decision.

1. **Artifact layer.** `Reader` opens the `.ninfer` file; `is_deepseek_v41_artifact` checks the extension and the `text` component and rejects an Engram-table artifact passed as the model (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:76-90`).
2. **Model layer** (`src/models/deepseek_v41`). `load_model` binds each tensor with a residency. Embedding, head and final norm are device-resident (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:158-172`). Routed experts become `Pinned` for Host and `Values` for Disk when offloading, and are device-resident otherwise (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:258-275`). It then produces one `ExpertBlob` pair per expert (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:388-404`).
3. **Executor layer** (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp`, 3100 lines by `wc -l`). It plans memory, builds the `ExpertDelivery`, runs the layer loop, and owns CUDA graphs (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2326-2358`, `:2627-2644`).
4. **Engine core.** `DeepSeekV41Core` validates options, runs the startup phases, and runs a worker `loop` that admits requests and steps them (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:145-215`, `:271-293`, `:974-1000`).

The expert transport is not DeepSeek-specific. `ExpertDelivery` is a core service shared with other families: the header describes it in family-neutral terms (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:176-201`), the tier is documented as shared by "every MoE family with host or disk experts" (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:3-5`), and the CPU path is an interface the family implements (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:48-56`). DeepSeek supplies the implementation `CpuExperts` (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:872-881`).

Pattern takeaway: the seam a port should use is the pair of `ExpertBlob` sources (model layer) and `ExpertCpuProducts` (family layer). Everything between them is policy-free transport.

## 3. Memory placement on a 32 GB WDDM card

What sits where when the Coder runs with offloaded experts:

- **VRAM, fixed at load.** The token embedding, output head and final norm are bound through `Bindings` (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:168-172`), whose `parameter` and `direct` default to `Residency::Device` (`G:\REPO_CLONES\ninfer-all\src\models\qwen3_5\load\bindings.h:41-48`), as are the per-layer hyper-connection weights (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:181-190`); the rest of the per-layer dense binding was not re-read here. The Engram projection and gate are device resident on Engram blocks only (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:276-285`). The docs give 6.0 GiB of dense weights for the full model (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:190-191`).
- **VRAM, planned after weights.** `free_after_weights` is measured from `cudaMemGetInfo` (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:246-253`). The executor then checks the KV caches for every sequence at full context against free memory and refuses with numbers (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:455-484`), checks the workspace plus a 320 MiB CUDA-graph reserve (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:607-618`, `:987`), and only then plans expert delivery (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:619-621`).
- **VRAM, the expert cache takes the rest.** In the default mode the cache is elastic: `growing` is true when there are expert banks, no explicit cache size and VMM granularity is available (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:400-404`). The cache reserves addresses for every expert and backs only the slots it has, through `GrowableRegion` (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:669-679`, `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:100-103`). The KV caches grow on demand and take memory back from the cache (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1083-1108`), and a prompt span borrows cache memory while it runs (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1110-1127`).
- **Layer banks.** At least one whole-layer bank must fit within `free - keep_free_bytes`, otherwise the construction throws (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:606-614`). Two banks are the default; one is the fallback (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:91-94`).
- **Pinned host RAM, Host residency.** `materialize` creates a `HostPages` block sized to the plan (`G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:808-813`), reads the expert bytes into it with direct reads on several threads (`G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:209-212`, `:850-853`), and only then pins it (`G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:876-877`). Expert blobs are pointers into that block (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:96-99`). On this PC that block would be 61.50 GiB (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:158`).
- **Disk residency.** The experts stay in the artifact's files (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:416-421`). Delivery opens the files, allocates two pinned staging halves (128 MiB by default) and a `FileReadQueue` (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:565-576`, `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:90`). On Windows that queue uses overlapped `ReadFile` on an IO completion port (`G:\REPO_CLONES\ninfer-all\src\core\file_read_queue.cpp:95-112`, `:190-202`). The docs state that no expert file is mapped (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:210-213`) and that host pinned memory is about 138 MiB (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:193-194`).
- **Engram table.** Not part of the weights load: it is read per pass (see section 4), by default from disk with buffered IO (`G:\REPO_CLONES\ninfer-all\include\ninfer\engine_options.h:423-427`).
- **Host-KV tier.** With prefix reuse on, the core also creates a pinned `HostPages` block for host KV (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:712-728`). The fork notes call out that this default is 8 GiB on top of the experts (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:168-171`).

WDDM-specific facts the code states:

- A pinned host allocation on Windows is "mapped into the GPU's address space and charged against the card" per the error-text comment (`G:\REPO_CLONES\ninfer-all\src\core\arena.cu:416-424`). The fork notes mark the effect on a 61.5 GiB pin as unmeasured (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:162-166`).
- `HostPages` compares the request to available physical memory first (`G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:41-45`), using `GlobalMemoryStatusEx` (`G:\REPO_CLONES\ninfer-all\src\core\host_memory.cpp:150-154`).
- `--wddm-evictable-budget` exists as a serve flag (`G:\REPO_CLONES\ninfer-all\src\serve\serve_options.cpp:1464-1466`); its effect is that every owning arena commits its pages at creation (`G:\REPO_CLONES\ninfer-all\src\core\arena.h:45-50`). Whether it is useful for the Coder is unclear from evidence.

Pattern takeaway: NInfer's memory plan is a sequence of "check against free VRAM, refuse with numbers" gates, with the expert cache as the elastic remainder. Both KV growth and span workspace take memory back from the cache. A Strata port that sizes its VRAM expert cache once would lose this elasticity.

## 4. Lifecycle and data flow

### 4.1 Startup

1. `ninfer-serve` binds its port first and starts answering 503 while weights load (`G:\REPO_CLONES\ninfer-all\apps\serve\main.cpp:295-306`); every route returns `model_loading` with `Retry-After: 2` until the ready flag is set (`G:\REPO_CLONES\ninfer-all\src\serve\http_server.cpp:471-482`).
2. `Engine::Impl` selects the DeepSeek core by artifact (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\engine.cpp:459-471`).
3. Option validation runs before any load: BF16 KV only, no Flash-Next hybrid options, residency rules for DMA share and CPU threads, one device unless tensor-parallel (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:145-175`).
4. The Engram table layout is resolved before the weights load, so a missing table fails at once (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:208-215`).
5. `load_model` and `materialize` place weights (section 3), then the core records free VRAM (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:217-225`, `:246-253`).
6. The executor is constructed with the prefill chunk at 2048, retrying 1024 then 512 on `InsufficientMemory` (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:321-333`).
7. `warm_up` runs one prompt forward, every decode batch size eagerly and then as captured CUDA graphs, then drains delivery and primes the cache from the routing profile (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2793-2855`). The comment explains why: a kernel loaded lazily while a graph waits on the expert host service would deadlock (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2794-2798`).
8. Startup diagnostics log where experts live and whether the cache was seeded "from the built-in profile", "from --expert-profile" or "by the first requests" (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:397-418`).

Cache seeding on the Coder: the built-in profile matches only when family and every layer's width match (`G:\REPO_CLONES\ninfer-all\src\models\expert_profile.cpp:67-75`); the Coder's widths are 192, so the fork notes say it matches nothing and the cache fills from use (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:55-60`, `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:308-317`).

### 4.2 One decode step

1. The worker `loop` waits for work, then calls `admit()`, `step()` and `check_progress()` inside one `try`; any exception that reaches its `catch (...)` goes to `fail_engine` (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:974-1000`, `fail_engine` at `:956-972`).
2. `decode` validates the batch, calls `back` to grow KV, stages tokens, then `read_engram_rows` on the host, then runs a graph keyed by batch size (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2599-2625`). `run_graph` runs the first call eagerly and later captures and launches a CUDA graph (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2627-2644`).
3. `run` loops over layers: Engram block when the layer has one, hyper-connection mixes, attention, mixes, FFN (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2348-2357`).
4. In the FFN, the router GEMM and route kernel pick experts (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1996-2004`), the routes are sorted (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2058-2060`), and if the layer has a delivery, `delivery->begin` is called (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2064-2073`).
5. `begin` launches the `split_routes` kernel on the compute stream (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1955-1962`, `:1994-1999`). The kernel counts routed experts, treats the call as wide when routed experts are half the layer or more, marks each routed expert as a cache hit or a miss against the cache's address range (and the lent banks' ranges for a narrow call), and publishes the request to the host service thread (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:155-162`, `:169-173`, `:237`).
6. The vector path computes the cached experts first, then the shared expert while misses arrive, then awaits delivery and computes the missing experts; CPU-computed products are added in a fixed-point sum (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2105-2122`). The wide path runs shared, awaits, then uses matrix products (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2131-2140`).
7. Sampling follows decode; after the sampled tokens are copied and the stream is synchronized, the core calls `check_experts` to surface any stored delivery failure (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:1499-1502`; `check_experts` itself at `G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2986-2990`).

The overlap in step 6 is the performance idea of the whole transport: GPU compute for cached experts and the shared expert runs while the host service thread loads misses. The service thread spins for 500 ms after the last request and then sleeps in 20 microsecond steps (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1780-1795`, `:1812`).

### 4.3 Engram lookup (host side, before the GPU pass)

- `read_engram` hashes each token's n-gram history into row ids, lays them out by block, waits for the previous copy event, reads the rows from the table reader, copies them to the device and records an event (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1441-1469`).
- The hash is a multiply-XOR rolling window per head: `layer_base + offsets + rolling % primes` (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:190-201`).
- The read is synchronous from the executor's view: `read_rows` is `submit` then `wait` (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:944-948`).
- On the GPU, `engram_block` decodes the IQ4_NL rows, projects them and gates them into the residual (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1524-1539`).
- Residency choices: `disk` (default, optional row cache), `ram` (whole table, 103 GiB), and `ram-hot` which DeepSeek refuses (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:642-648`, `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:184-188`). On Windows, files open with `FILE_FLAG_RANDOM_ACCESS`, plus `NO_BUFFERING` for direct IO and a mapping for mapped IO (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:100-112`), and `--ngram-lock` uses `VirtualLock` (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:368-376`).
- Separately-stored table: the model names it by digest; `--ngram-table PATH` supplies it, and a digest or format mismatch throws (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:113-140`). For the Coder this is a 110.6 GB file (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:115-118`), so on this PC it is read in rows, one pass at a time, through the Windows file cache.

## 5. Expert delivery and tier (where Strata is already present)

This is the section a Strata port most needs, because NInfer states its debt to Strata in the source.

**What NInfer says it took from Strata**
- The expert tier "(frequency counts, rounds and their measured constants)" and the sparse-attention KV residency map follow Strata's designs, Strata's transcription "served as a reference for the n-gram hash, the expert cache and its measured pitfalls", and Strata's CPU-computed expert misses "led to the CPU share". The sentence is in the Qwen3.8-Flash-Next bullet (`G:\REPO_CLONES\ninfer-all\CONTRIBUTORS.md:74-81`).
- The tier header says Strata measured its constants on Qwen3.8-Flash-Next: a round every four verification windows, up to 96 swaps, decay 0.7, gain 1.5 (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:20-21`). Those are the defaults in `ExpertTierOptions` (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:31-37`).
- DeepSeek-V4.1 support was "measured against Strata's `strata-dsv41`" (`G:\REPO_CLONES\ninfer-all\CONTRIBUTORS.md:82-85`).
- Strata's own source is not in this repo; what is known about it here comes from these comments and the docs. Anything beyond that is unclear from evidence.

**Tier policy (as coded in the header comment)**
- Each narrow call adds one to the count of every routed (token, expert) pair. Every `every` decode passes a round swaps the most routed experts the cache lacks into the slots of the least routed, at most `swaps`, each only if its count beats the victim's by `min_gain` and reaches `min_count`; then all counts are multiplied by `decay` (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:3-10`).
- A miss the CPU computes never moves expert bytes; it can enter the cache only through a round (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:10-12`).
- The tier keeps no slots and copies nothing; the owner of the slots applies swaps (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:14-18`).

**Delivery policy (as coded in the header comment)**
- A narrow call sends a share of misses to the CPU where the family supports it; the others enter a free slot, else the coldest unrouted expert's slot if the tier admits them, else a layer bank (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:181-188`).
- A wide call (a prompt chunk) fills a layer bank, evicts nothing and counts nothing, "so a prompt does not push out the decode working set"; it also promotes its most routed experts into free slots (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:189-195`).
- Between wide calls the banks are lent to the cache as extra slots and taken back by the next wide call (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:95-98`, `:196-201`, `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1570-1583`).
- The next layer's bank is prefetched while the current layer computes; the successor is learned from consecutive wide calls (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1562-1567`, `G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:91-94`).
- Copies are coalesced (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1050-1055`); pinned pieces go by `cudaMemcpyAsync`, disk pieces through staged file reads (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1136-1150`).

**CPU share (the Strata-inspired piece)**
- It is built only if the CPU can compute these expert formats and `cpu_share` holds. `cpu_share` is true when the user sets a DMA share below 1, or by default only for Host residency (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:864-872`). The code comment records that on an RTX 4090 host every split of disk experts decoded slower than copying all misses (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:864-866`).
- The CPU path reproduces the GPU's integer sums, so tokens do not depend on the split (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:235-242`).
- Doc claim on the comparison: on a rented RTX 4090 + EPYC host with host experts, Strata computed 61% and 53% of its expert products on CPU, while NInfer's cache served 62 to 72% and its CPU share computed 16 to 22% (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:539-543`). NInfer's doc states "Strata copies no missing expert: its CPU threads compute them" (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:564`). Those runs were on 128 threads and 409 GiB of RAM (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:510-512`), a very different box from Ryan's 16-core, 96 GB machine; the numbers do not transfer.

Pattern takeaway: the whole expert path is a three-way choice per miss: copy over the bus, compute on the CPU, or already cached. NInfer makes that choice per call with measured copy and CPU speeds (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:878-880`, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:230-232`).

## 6. Configuration and API surface

**CLI flags that decide the Coder's behavior** (parsed in `ninfer-serve`; whether the CLI and perplexity apps share this parsing is unclear from evidence, not opened):
- `--expert-residency device|host|disk` and `--expert-cache-mib N|auto` (`G:\REPO_CLONES\ninfer-all\src\serve\serve_options.cpp:1137-1159`).
- Hybrid expert options are delegated to a shared parser (`G:\REPO_CLONES\ninfer-all\src\serve\serve_options.cpp:1160`); DeepSeek rejects the Qwen-Flash-Next-only ones (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:149-154`).
- Engram flags are validated in the core (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:176-197`).
- Drafting is `--spec dspark` or `--lookup-ngram` only (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:118-144`), and `--spec dspark` needs an artifact converted with the drafter (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:200-203`).
- Planned first-run flags for this PC: `--expert-residency disk --no-prefix-reuse`; never `--ngram-residency ram`, `--ngram-io mmap`, `--host-cache-mib auto` or `--disk-kv-path` (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:173-176`). A Host run is deferred until a small `cudaHostRegister` probe ladder (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:174-176`).

**Per-model data, not flags.** The model's identity and Engram table are artifact data: the table is named by SHA-256 digest and checked against the separate table artifact (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\engram.cpp:113-140`). Routing profiles are written as JSON with a version, an artifact id and per-layer counts (`G:\REPO_CLONES\ninfer-all\src\models\expert_profile.cpp:60`).

**Server behavior.** Startup returns exit code 1 for startup errors and 2 when the engine fails after serving, so a supervisor can reload (`G:\REPO_CLONES\ninfer-all\apps\serve\main.cpp:425-440`). An engine watch polls `has_failed()` every 250 ms and stops the server on failure (`G:\REPO_CLONES\ninfer-all\src\serve\http_server.cpp:1547-1567`). The wider HTTP surface (OpenAI chat/responses, Anthropic messages, llama.cpp-style endpoints) is described by the scout; its route table was not re-opened for this study, so it is unclear from evidence at file:line level beyond the readiness gate above.

Pattern takeaway: the user-facing surface is almost entirely flags plus artifact data; a Strata port can keep the same flags (`--expert-residency`, `--expert-cache-mib`, a profile file) and put its own policy behind them.

## 7. Error handling

Described as behavior, not as findings.

- **Fail early and loudly at load.** Option conflicts throw `invalid_argument` before any load (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:145-197`). A missing Engram table fails before weights are read (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:208-215`). KV and workspace checks throw `InsufficientMemory` with MiB figures (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:475-483`, `:610-617`).
- **One recovery path at startup.** Only `InsufficientMemory` while constructing the executor is caught, to retry a smaller prefill chunk (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:324-333`).
- **Host pin is fail-fast.** `HostPages` throws "failed to pin N MiB" when asked for more than available RAM, and again if CUDA registration fails (`G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:41-45`, `:93-98`). The fork notes state there is no fallback if the pin is refused (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:168-171`).
- **Per-request vs engine-wide.** Only `RequestError`, `invalid_argument` and `GrammarViolation` count as request errors (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:62-72`). `fail` rethrows anything else (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:1032-1043`), which reaches `fail_engine`: the engine logs, marks itself failed, stops, and fails running requests (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:956-972`). `main` then exits with status 2 (`G:\REPO_CLONES\ninfer-all\apps\serve\main.cpp:432-435`).
- **Runtime memory pressure.** KV growth first shrinks the expert cache; if nothing is left it throws `InsufficientMemory` (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1099-1105`). Span borrowing degrades gracefully instead: it returns empty when the cache cannot give more (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1113-1127`), and `forward_span` then runs the span as two halves of whole chunks, down to chunk by chunk (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2481-2493`). The span size itself is chosen at startup by `plan_span`, which tries 8, 4, then 2 chunks (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1015-1022`).
- **Delivery failures are latched.** The host service thread stores the first exception, drops queued work and neutralizes the call's output so a captured graph can finish; the caller checks before publishing (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1758-1778`). The loop wraps `serve` in try/catch and still signals `prepared` (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:1803-1806`), and `check_experts` rethrows (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2986-2990`).
- **CUDA errors abort.** `CUDA_CHECK` prints and calls `std::abort()` (`G:\REPO_CLONES\ninfer-all\src\core\device_launch.cpp:15-20`), so a raw CUDA failure on these paths kills the process.

Pattern takeaway: NInfer's model is "refuse with numbers before running, latch and rethrow after". A Strata port should keep that split, and should decide up front which of its own failures (a CPU worker fault, a file read error) are per-request.

## 8. Performance techniques worth copying

- **Direct reads for the weight load** so the OS page cache does not keep a second copy of what the process pins (`G:\REPO_CLONES\ninfer-all\src\artifact\materializer.cpp:209-212`).
- **Multithreaded page touch before registering** a big pinned block (`G:\REPO_CLONES\ninfer-all\src\core\host_pages.cpp:70-81`).
- **No repacking**: experts are served from the stored GGUF rows (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\model.cpp:65-66`).
- **Overlap cached compute, shared expert and miss arrival** (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2105-2122`).
- **Eager warm-up of every shape, then graph capture** (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2793-2855`).
- **Prompt chunk 2048** because the wide matrix kernel fills its widest tile only at wide chunks (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:321-324`), plus multi-chunk prompt spans sized against free memory (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1004-1024`).
- **Windows file IO**: overlapped IOCP reads for expert files (`G:\REPO_CLONES\ninfer-all\src\core\file_read_queue.cpp:95-112`) and random-access flags for the Engram table (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:100-112`).
- **Avoid memory-mapping large files on Windows**: the fork's own conversion run found that mapped pages stay in the working set and drove system RAM from 13% to 98%; the notes say never to pass `--ngram-io mmap` (doc claim, `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:136-141`).

Reported speeds are doc claims from other hardware (RTX 4090, EPYC, 188 GiB+ or 409 GiB RAM) and none were measured on this PC (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:207`, `:510-512`, `G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:162-166`). The one RTX 5090 figure in the docs is 48.5 to 49.6 tok/s on a PCIe 5.0 host with a cache of 15% of the experts, for the full model (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:562-564`); it is not the Coder and not this host.

## 9. Strata port design notes (adaptation guidance)

The guiding finding: NInfer already contains a Strata-derived tier and a Strata-inspired CPU share. A port should treat NInfer as the host engine and ask which Strata behavior is still missing, not re-add what exists (`G:\REPO_CLONES\ninfer-all\CONTRIBUTORS.md:74-81`).

- **Pattern takeaway for the expert tier:** the tier is a separate policy object that counts routes, decays, and returns swaps, while the cache applies them (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:14-18`). To plug in a different Strata policy (other constants, a different admission rule), replace `ExpertTier` behind its options (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:31-37`) and keep `ExpertDelivery`'s slot ownership (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.cu:2109-2115` shows `prime` resetting the tier). Strata's constants were measured on a different model (Qwen3.8-Flash-Next); whether they hold for a 192-expert Coder on a 5090 is unclear from evidence (`G:\REPO_CLONES\ninfer-all\src\core\expert_tier.h:20-21`).
- **Pattern takeaway for the routing profile:** the cache can be primed at the end of warm-up from a profile (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2837-2842`), but the built-in profile is matched on exact per-layer widths (`G:\REPO_CLONES\ninfer-all\src\models\expert_profile.cpp:67-75`), so the Coder gets none (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:55-60`). A Strata routing profile (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:520-522`) would have to be remapped through the Coder's kept-experts list, which the fork notes say is "not done" (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:58-60`). The profile hook is `--expert-profile`, read at `G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:311-313`. Whether Strata's profile is 384-wide is unclear from evidence.
- **Pattern takeaway for CPU-computed misses:** the plug-in point is `ExpertCpuProducts` (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:48-56`), wired in `plan_delivery` (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:872-881`). Two constraints from the evidence: the CPU result must be bit-identical to the GPU's fixed-point sums, which is how NInfer keeps tokens independent of the split (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:241-242`), and for Disk residency NInfer defaults to copying everything (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:864-871`). On Ryan's 16-core Ryzen the first-run plan is Disk, so a CPU share would not engage by default. Whether it helps on this CPU is unclear from evidence (the CPU-share runs read for this study used 61 CPU threads by default on a 128-thread host, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:510-517`; the other 4090 host's thread count is not stated in the lines read).
- **Pattern takeaway for the VRAM budget:** the cache is the elastic remainder after KV, workspace and graph reserve, and it yields to KV growth and span borrowing (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1083-1127`). Any Strata cache sizing should plug into `keep_free_bytes` and `elastic` (`G:\REPO_CLONES\ninfer-all\src\core\expert_delivery.h:89`, `:100-103`) rather than a fixed VRAM fraction. On WDDM the headroom matters more because pinned host memory also counts against the card (`G:\REPO_CLONES\ninfer-all\src\core\arena.cu:416-424`).
- **Pattern takeaway for host vs disk experts on this PC:** Host pins 61.5 GiB of 96 GB and is untested under WDDM; Disk pins about 138 MiB and relies on the OS page cache (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:158-166`). A Strata engine that memory-maps experts (its `--mmap-experts` setting appears in the doc's flag list, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:536`) would be in the class of risk the fork has already hit on Windows (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:136-141`); NInfer's Disk mode deliberately maps no expert file (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:211-213`). Copy that choice.
- **Pattern takeaway for expert-skipping tricks:** Strata's `--experts 5` keeps 5 of 6 experts and, per NInfer's doc, "computes another model" (`G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:524`). NInfer's DeepSeek core refuses tail skip and similar approximations outright (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:149-154`). A port that wants such a mode must add it as an explicit, labeled option; it cannot reuse the exact-token path.
- **Pattern takeaway for speculative decoding:** NInfer accepts only the release's DSpark drafter or n-gram lookup for DeepSeek (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:118-144`). For the Coder the drafter source is not on this machine and is "unproven against this backbone" because it was built for the unpruned model (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:183-189`). Strata's drafter confidence setting is already mirrored (`--draft-confidence` 0.5 default, per the doc line `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:305`).
- **Pattern takeaway for the prefill path:** NInfer runs a long prompt as one span against free VRAM (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:1004-1024`), the doc credits this with 1.36 to 1.48 times Strata's prefill rate on the 4090 host (doc claim, `G:\REPO_CLONES\ninfer-all\docs\deepseek-v41.md:548-551`). A Strata port gains nothing by replacing it; the plug-in work is on decode.
- **Pattern takeaway for the Engram table:** it is its own reader abstraction (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.h:52-70`) with disk, RAM and row-cache modes (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:642-648`). The read is synchronous per pass (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:944-948`) and runs on the host before the GPU pass (`G:\REPO_CLONES\ninfer-all\src\models\deepseek_v41\executor.cpp:2619`). On this PC the table is 110.6 GB against 96 GB RAM, so it cannot be held (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:117-118`, `:160-161`); any Strata mechanism for it (prefetch, a hot-row cache) plugs into the reader, and the existing hot-row mode is refused for DeepSeek (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:184-188`). The reader already has a `prefetch(row_ids)` hint that skips rows a cache or hot slot holds (`G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.h:91`, `G:\REPO_CLONES\ninfer-all\src\models\qwen4_exp\ngram_table.cpp:950-961`), and a grep of `src/models/deepseek_v41/*.cpp` finds no call to it, so the DeepSeek path does not use it. Whether prefetching hash rows one token ahead is possible is unclear from evidence, because the row ids depend on the sampled token.
- **Pattern takeaway for build and run on this PC:** build for `120a` with MSVC 14.44 and CUDA 13.1+ (or 13.0 with the fork option), accept the sm8x compat kernels as default (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:77-93`, `:105-115`), and expect no DirectStorage (`G:\REPO_CLONES\ninfer-all\CMakeLists.txt:187-194`). Any Strata CUDA code added would need to build under the same constraints.
- **Pattern takeaway for what to measure first:** the Coder has not run on this fork (`G:\REPO_CLONES\ninfer-all\docs\fork\CODER.md:10-12`), so the first numbers should be the startup diagnostics (`G:\REPO_CLONES\ninfer-all\src\runtime\engine\deepseek_v41_core.cpp:397-418`) and the delivery stats counters, before any Strata trick is blamed or credited. Which counters are exposed beyond those logs is unclear from evidence.
