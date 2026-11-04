# Running ISTA-DASLab's DeepSeek-V4.1-Flash GSQ-RCO Coder on this fork

The Coder is at `G:\Models\litesuite\DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder`:

- `IQ2_XXS\...-0000{1..5}-of-00006.gguf`: shard 1 is metadata only (5.78 MB); shards 2-5 hold the
  transformer, 72,118,... bytes (72.1 GB), 192 of the 384 routed experts per layer;
- shard 6 is `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110,595,290,816 bytes), hard-linked;
- `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` (972 MB), `expert-selection\...kept-experts.json`.

**Nothing below was converted, loaded or run.** Everything marked "checked" was checked from file
headers and a header-level dry run of the converter (`build_model` and the recipe, with the 103 GiB
digest stubbed out), which reads no weights.

## (a) Does the converter accept the Coder, and how does the router map to the kept experts?

**Not as upstream shipped it. It does now, with two things done by hand and one patch (C1).**

Three incompatibilities, all found by running `GGUFShards` / `validate` / `build_model` on the header:

1. **Expert count.** `validate` requires `deepseek41.expert_count == config["n_routed_experts"]`
   (`tools/convert/deepseek_v41.py:286`, base `:247`) and `moe_block` sizes the router, bias, gate/up and
   down banks from `config["n_routed_experts"]` (`moe_block`, `:840`, base `:771-777`). The Coder's GGUF says
   `expert_count = 192` and every `blk.N.ffn_{gate,up,down}_exps.weight` is `(192, ...)`. The release's
   `config.json` says 384. **Fix without code: convert from a copy of `config.json` with
   `text_config.n_routed_experts = 192`.** That copy is at
   `G:\REPO_CLONES\ninfer-all-work\coder-hf\config.json` (made from `F:\AI_Models\DeepSeek-V4.1-Flash-1017\engram\config.json`,
   3,311 bytes, the only change is that one number). The C++ side reads the same key from the artifact
   (`src/models/deepseek_v41/config.cpp:108`, `:192`) and nothing in the model or the Engine core assumes
   384: the router top-k kernel accepts up to 512 (`src/models/deepseek_v41/kernels.cu:1031`, `:1508`).
2. **Metadata names (change C1).** ISTA's files written for llama.cpp b11870+ call the Engram keys
   `deepseek41.ple.*` / `deepseek41.engram.*` and the indexer keys `attention.indexer.*`, where the recipe
   was written against `engram_*`, `kv_source_layer_ids`, `candidate_*`. `kv_view` now fills the old names
   from the new ones (`tools/convert/deepseek_v41.py:229-263`; `validate` uses it at `:269`). The hash
   constants are metadata arrays, not tensors: `engram_constants` builds them from `engram.token_map`,
   `ple.layer_multipliers` (8 = 2 blocks x 4), `ple.head_vocab_sizes` (the primes, 48 = 2 x 24) and
   `ple.head_offsets` (`:518-560`). `head_offsets` counts rows from the start of the whole table; the model
   wants them from the start of the block (`src/models/deepseek_v41/engram.cpp:200`,
   `layer_base_[l] + offsets[l][c] + hash % primes[l][c]`), so the second block's are rebased. The per-block sums of the
   primes are 384,006,168 and 384,016,682, exactly `engram_num_embeddings` of the release's config.
3. **Tensor names (change C1).** The Engram block's `engram_wkv` is `engram_kv` here (25600 x 6144 Q5_K,
   the same shape the recipe expects), and the gate is stored as the product, `engram_gate.weight`
   (4 x 5120, F32), where the recipe multiplied `engram_q` by `engram_k` (`:872-883`, base `:802-806`).
   The table is **one** tensor `per_layer_token_embd.weight` (768,022,850 x 256 IQ4_NL) in a file with no
   `general.architecture`, where the recipe read one `blk.N.engram_embd.weight` per block from a
   `deepseek41-engram` file (`engram_table_source` `:695-720`, base `:642-656`; `table_digest` `:729`).
   The rows are in block order, so the artifact's table is the same bytes.

**Router to kept experts.** There is no mapping at conversion or run time. The GGUF stores the router
already cut to the kept experts: `blk.N.ffn_gate_inp.weight (192, 5120)`, `exp_probs_b.bias (192,)`,
`exp_probs_b_vl.bias (192,)`, and the expert banks `(192, 2304, 5120)`. Expert `j` of layer `L` is the
base model's expert `kept_experts[L][j]` (README "How the Experts Are Chosen";
`expert-selection\...kept-experts.json` has 40 layers x 192 sorted ids, checked). The converter reads
tensors positionally, the engine sees a 192-expert MoE and takes the top 6 of 192 (renormalised), which
is what RCO optimised. `deepseek41.hash_layer_count = 0`: no layer routes by token hash.
The kept-experts list matters in one place only: the built-in routing profile
(`src/models/expert_profiles/builtin.inc`, family `deepseek-v4.1`) is 384-wide, so
`builtin_expert_profile` (`src/models/expert_profile.cpp:67`) finds no match for 40 x 192 and returns
nothing (`src/runtime/engine/deepseek_v41_core.cpp:315`); the expert cache then fills from use instead of from
a profile. A profile remapped through the list would need writing (not done); it is a warm-up speed-up, not
a correctness matter.

**Checked** (`python -I` with the repo's own parser, no network): `validate` OK, `engram_constants` OK,
`engram_table_source` OK (768,022,850 x 256, `gguf_iq4_nl`), `build_model(text)` OK (962 parameters, router
`(192, 5120)`, `gate_up (884736, 5120)`, `down (983040, 2304)`), `build_model(text, vision)` OK (1261
parameters, `image_router_bias (40, 192)`), `deepseek_v41_gguf` recipe with `text,engram` OK with the
digest stubbed. `tests/convert/test_deepseek_v41.py`: 21 passed.

## (b) Can `--components engram` build the Engram table from our GGUF, locally?

**Yes.** `--components engram` needs only `--source engram=<the Engram GGUF>` and the config
(`configure_engram`, `tools/convert/deepseek_v41.py:1037`); `G:\...\DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`
is that file (same bytes as shard 6, hard-linked). It reads the table row for row (IQ4_NL stays IQ4_NL),
hashes it (SHA-256 over 110.6 GB, which the model records), and writes it. No 110 GB download.

## (c) The exact conversion commands, sizes and where they go

Run from `G:\REPO_CLONES\ninfer-all` with the Python 3.11 that has numpy and torch (CPU is enough:
`--device cpu`). **Not run.**

Still missing, and it is a download (Ryan's go): the release's **`tokenizer.json` and
`tokenizer_config.json`** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, a few MB) in the HF directory.
`_load_resources` (`:573`) reads them from `--model` and the repo carries only the chat template and
generation config (`tools/convert/templates/deepseek_v41`). The GGUF holds the tokens and merges but not the
pre-tokenizer, so rebuilding `tokenizer.json` from it would be a guess. Put the two files in
`G:\REPO_CLONES\ninfer-all-work\coder-hf\`.

```bat
set HF=G:\REPO_CLONES\ninfer-all-work\coder-hf
set CODER=G:\Models\litesuite\DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder
set ENGRAM=%CODER%\DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf

rem 1. the vision tower's routing bias, from the Coder GGUF (already done: ninfer-all-work\bias_vl.safetensors, 40 x 192 F32)
python -m tools.convert.bias_vl_from_gguf %CODER%\IQ2_XXS\DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf --out G:\REPO_CLONES\ninfer-all-work\bias_vl.safetensors

rem 2. the model: text + vision (no drafter), its Engram table named by digest
python -m tools.convert --model %HF% --recipe deepseek_v41_gguf --components text,vision ^
  --source gguf=%CODER%\IQ2_XXS\DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf ^
  --source engram=%ENGRAM% ^
  --source vision=%CODER%\mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf ^
  --source bias_vl=G:\REPO_CLONES\ninfer-all-work\bias_vl.safetensors ^
  --name DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS ^
  --out C:\Users\Ryan\.litesuite\llm\ninfer-models\DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-ninfer-v3.ninfer --device cpu

rem 3. the Engram table, once
python -m tools.convert --model %HF% --recipe deepseek_v41_gguf --components engram ^
  --source engram=%ENGRAM% --name DeepSeek-V4.1-Flash-Engram-IQ4_NL ^
  --out G:\Models\litesuite\ninfer\DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer --device cpu
```

Both runs read the whole 110.6 GB Engram GGUF once to hash it (the model records the digest, and the run refuses
a table with another one). Upstream's measured times: 6 min 23 s for the 125.8 GB model on 96 threads,
3 min 29 s for the table on 24 cores (`docs/deepseek-v41.md:144`); expect longer on this disk.

**Sizes** (from the headers): routed experts 66,036,695,040 bytes (61.50 GiB); everything else in shards 2-5,
6,081,024,448 bytes (5.66 GiB); mmproj 0.97 GB. **Model artifact: about 73 GB** (shards 2-5 are 72.1 GB; the
vision tower adds 0.9 GiB; plus the artifact's directory and alignment). **Table artifact: about 110.6 GB**
(110,595,290,400 bytes of rows; upstream's is 110,595,294,496).

**Where** (free space now: C: 95 GB, G: 68 GB, D: 104, E: 100, F: 116, H: 219; F, H and D share one USB hub, and
F is storage only):

| step | writes | fits |
|---|---|---|
| 2. model | `C:\Users\Ryan\.litesuite\llm\ninfer-models\` | C: has 95 GB; leaves about 22 GB. G: has 68 GB, not enough |
| 3. table | `G:\Models\litesuite\ninfer\` | **no drive that is not USB has 111 GB free.** G: has 68 GB; after step 2 the Coder's shards 2-5 and the mmproj (73 GB) can be deleted (they are re-downloadable and the model artifact holds the same blocks), which gives G: 141 GB. Only then does the table fit, with its source still on G: |

So the order is 2, verify the model artifact, delete shards 2-5 and the mmproj **with Ryan's go**, then 3, then the
Engram GGUF (and shard 6, the same inode) can go too, which frees 110.6 GB again. The other route is writing the
table to H: (219 GB free) and moving it; H: is on the shared USB hub, so that needs Sentinel's clearance first.

## (d) RAM

`--expert-residency host` pins **61.50 GiB** of routed experts (66,036,695,040 bytes; per layer 1,418 MiB, layer 39
1,873 MiB). Ryan's box has 96 GB (about 89 GiB usable). That leaves about 27 GiB for Windows, the engine's other
pinned buffers, the CPU-share threads and the Engram table's page cache (the table is read a row at a time; the
default `--ngram-residency disk`). `--ngram-residency ram` cannot fit (103 GiB). **By arithmetic it fits.**
Two Windows caveats the upstream docs state and nobody has measured with this model: "WDDM charges pinned host
memory against the card's memory" (`docs/build-windows.md:87`, `docs/maintainer/launcher-profiles.md:73`), and no
Windows number for DeepSeek exists at all (its measurements are Linux, 188 GiB hosts, and "a host with 128 GB ... not
tested", `docs/deepseek-v41.md:207`). If pinning 61.5 GiB is refused (`failed to pin`), `--expert-residency disk`
keeps only 138 MiB pinned and serves the experts from the OS page cache (`docs/deepseek-v41.md:193`).

## Optional components

- **vision**: needs nothing more than step 2 already has. The mmproj is local; the routing bias is built from the
  Coder GGUF's `exp_probs_b_vl` (`tools/convert/bias_vl_from_gguf.py`) instead of the base release's 48 shards
  (the old route, `fetch_tensors`, was the network path N20). Run with `--vision`.
- **dspark drafter** (`--components text,vision,dspark`, `--spec dspark`): needs a drafter source that is **not on
  this machine**: Unsloth's `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`, about
  8 GB in the artifact, 7.4 GiB of device memory) or beamster's STQ pack (`beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw`
  at revision `07fec972538639726b868f81e89d9d4f5005fe10`: `config.json`, the index and `model-mtp-*.safetensors`,
  3.85 GB, 3.58 GiB of device memory), both downloads. It is also **unproven against this backbone**: the drafter
  was built for the unpruned model and the Coder has half its experts, so its acceptance rate here is unknown. The
  drafter's weights come out of the expert cache.
