# Does this fork's `ninfer-serve` accept the Qwen artifact we already have?

`C:\Users\Ryan\.litesuite\llm\ninfer-models\qwen3_8_27b_nvfp4.ninfer` (23,719,715,844 bytes, Ryan's everyday 27B) and
`qwen3_5_0_8b.ninfer`. **By reading code and the artifact's header: yes, no reconversion is needed.** Not confirmed
by opening it with the engine, which is a model load and needs Ryan's go.

- **Container version.** Both files start with `NINFER\0\x03`; this tree writes and expects the same magic
  (`tools/artifact/framing.py:5`, `src/artifact/framing.h:9-12`). `artifact_summary` of the repo's own reader (a header
  read, `tools/artifact/inspect.py`) reports `version 3` for both and parses every object.
- **Formats and layouts.** The 27B holds `bf16 579, fp32 264, fp8_e4m3fn_row_bf16 146, int32 1, nvfp4 112, q4_g64_fp16 55,
  q5_g64_fp16 54, q6_g64_fp16 1, q8_g32_fp16 28` in the layouts `block_scale_k16_m128x4_v1, contiguous_le_v1,
  row_scale_v1, row_split_k128_v1`. Every one of those names, formats and layouts, is in `src/artifact/formats.cpp` (`:15-21`,
  `:45-52`; checked by grep) and the Python tables.
- **Components.** `text, vision, mtp, dflash2`. `--spec mtp` and `--spec dflash2` are in `--help` of the built server.
- **Age.** The artifact was converted by an engine at or before `Neroued/ninfer@5b4303c0` (2026-09-16). That commit is an ancestor of this
  tree. Of the 45 commits since (HEAD before this fork's edits) that touch `src/artifact`, `tools/artifact` or the Qwen converter, the one that
  changes the container is `76f0182f` (2026-09-25, "carry one NVFP4 divisor per source matrix in a stacked tensor"), and
  its message says "Absent means one word and the present layout exactly, so every existing artifact stays valid and no
  reader changes behaviour"; `b44d7c3a` fixes the tensor-parallel binding of the same field.
- **What is not shown.** Binding the Qwen config (`src/models/qwen3_5/...`) to this file and the GPU run. Both happen at
  the first load.
