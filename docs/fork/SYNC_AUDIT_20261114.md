# Network audit of the 2026-11-14 upstream sync (upstream master 40d913db into fork/audited)

The merge is `c6a560d1`: fork parent `8272aea4` (18 fork commits on `c37a053b`) and upstream
`iamwavecut/ninfer-all` master `40d913db`, 334 commits after `c37a053b`. Why it was merged is in the
merge commit's body: WaveCut's Edge0-145B artifact needs `72d1d8e04` (latent experts, `gguf_*`
formats) and `57395b20b` (any n-gram table with the same hash constants). This audit asks one
question about the code that came in: does anything new open a connection, download something,
send telemetry or start a process? The method is the one in [NETWORK_AUDIT.md](NETWORK_AUDIT.md).

Status: static analysis only. **UNVERIFIED by Ryan.** The dynamic proofs that NETWORK_AUDIT.md
section 6 lists as missing are still missing.

## 1. How to re-check (paste and run, from the tree's root)

```bash
python -I docs/fork/audit_network.py                 # file judgement: exit 0, 2606 files, 181 hits, 0 unexpected
python -I docs/fork/audit_sync_diff.py 8272aea4 HEAD # every ADDED line that matches: 37 lines in 8 files (section 3)
git diff --stat 8272aea4 HEAD -- third_party         # vendored code: llama-jinja and ggml-quants only (section 4)
```

`audit_network.py` judges each file, so a new hit inside a file it already allows passes unseen.
`audit_sync_diff.py` covers that gap for a sync: it prints each added line that matches the same
patterns, plus process launches, telemetry words and URLs, so a person judges every one.

## 2. What the merge removed, because upstream added it after the fork's cut

The fork's cut `f0e0280a` removed CI, the rented-host (pods) harness, containers and nix. Upstream
modified 35 of those files (the merge keeps them deleted) and added these, which the merge also
removes:

| File (upstream 40d913db) | What it does | Disposition |
|---|---|---|
| `.github/scripts/hf_client.py` | `from huggingface_hub import HfApi`: CI uploads and downloads of build caches and artifacts on Hugging Face | **removed** (it was the scanner's only 2 unexpected hits) |
| `.github/scripts/object-weights.py`, `object-weights.json` | CI build-shard weights for the cache scripts above | **removed** |
| `.github/scripts/quiet-upload-progress.cjs` | silences the upload progress bars of the script above | **removed** |
| `scripts/pods/attach_dflash2.py` | the rented-host harness, as the rest of `scripts/pods/` | **removed** |
| `tests/release/test_hf_client.py`, `test_compiler_cache_finish.py`, `test_cuda12_glibc_patch.py` | tests that import the removed `.github/scripts` | **removed** |
| `tests/pods/test_attach_dflash2.py` | the test of the removed `scripts/pods/attach_dflash2.py` (removed by the docs commit after the merge) | **removed** |

`.github/FUNDING.yml` and `.github/pull_request_template.md` stay, as on `fork/audited` (text only).

## 3. Every added line the diff scan found, judged

| File | Lines | Judgement |
|---|---|---|
| `README.md`, `CONTRIBUTORS.md` | 24 | links in prose (model cards, contributors' repositories, an `hf download` example for the reader). Nothing runs them. **kept** |
| `cmake/MediaSources.cmake` | 4 | `NINFER_FFMPEG_URLS` and `NINFER_DAV1D_URLS` now also list a Gentoo distfiles mirror. These are data: no `file(DOWNLOAD)`, `FetchContent` or `ExternalProject` reads `*_URLS` (`grep -rn "_URLS" cmake CMakeLists.txt` finds only the `set()` lines). The fork's cut N02 holds: the archives are placed by hand and checked against the pinned SHA-256. Linux only. **kept** |
| `kernel/ninfer-direct/` (new) | 3 | two Linux kernel modules for `--expert-direct` (an NVMe drive writes disk experts straight into GPU memory through PCI P2PDMA), with a GPL license URL and an `apt install` line in its README. No socket, netlink or network symbol (`grep -rnE "socket\|inet\|netlink\|sock_create" kernel/` is empty). Built only by hand on Linux with `make`; the Windows build never compiles it. **kept** |
| `src/models/deepseek_v41/dspark_ranking.h` | 1 | a comment that credits the source of a ranking profile (a GitHub URL). **kept** |
| `tests/convert/test_dspine.py`, `tests/models/dspine/runtime_check.py` | 5 | `subprocess.run` of the local converter and a local test driver binary. Nothing is fetched. **kept** |

## 4. Vendored code

`git diff --stat 8272aea4 HEAD -- third_party`: 11 files in `third_party/llama-jinja` (template
engine string and value types) and `third_party/ggml-quants` (quantization kernels). Their added
lines hold no `curl`, `socket`, `getaddrinfo`, `httplib::Client`, URL, `system(` or `popen(`.

## 5. The fork's runtime cuts, re-checked on the merge

The raw greps of NETWORK_AUDIT.md section 1 give the same answers on `c6a560d1` as on `8272aea4`:
- no `curl/curl.h`, `curl_easy` or `getaddrinfo` in `src`, `apps` or `include`;
- `httplib::Client` only in `src/serve/mcp_proxy.cpp`, still limited to loopback (N11);
- one listener in `src/serve/http_server.cpp`;
- no spdlog network sink;
- no `OPENSSL` or `CPPHTTPLIB_` define.

## 6. Not part of the network question, but changed by this sync

- **DSpine.** Upstream's own DSpine (a separate drafter artifact, `--spec dspine --draft-model`)
  replaced the fork's port, which carried the drafter inside the `dflash` component. An artifact
  that the fork's converter made at `76c35c85` is refused on this build: with `--spec dflash`,
  `src/models/qwen3_5/config.cpp` checks the DFlash config with `require_members`, and
  `src/artifact/schema.cpp` throws `unknown member dspine`. The refusal is loud. The text
  component alone still loads. The merge body lists every fork DSpine file that was dropped.
- **Windows fixes.** W2 (`static constexpr` for MSVC C3495) is carried into
  `causal_cache/small_t.cpp`. Upstream moved the other files where W2 lived into generated units, so
  the build shows whether W2 is needed again there. W4 (refuse MSVC 19.38) stays in
  `CMakeLists.txt`, beside upstream's new refusal of CUDA 13.2.0 and 13.2.1 for sm_120a.
- **Taken from upstream.** These replace the fork's own versions: artifact writes (no zero fill on
  Windows, a sync every 512 MiB) and GGUF reads (bounded reads, ISTA's llama.cpp b11870+ dialect).

## 7. Python tests on the merge

Command, from the tree's root. The `tests` module is pinned to the tree because an unrelated
`tests` package in this machine's site-packages shadows the namespace; `tests/test_structured_output_live.py`
is a script against a running server and is left out:

```bash
python -c "import sys,types,os; sys.path.insert(0,os.getcwd()); m=types.ModuleType('tests'); m.__path__=[os.path.join(os.getcwd(),'tests')]; sys.modules['tests']=m; import pytest; sys.exit(pytest.main(['tests','-q','-p','no:cacheprovider','--ignore=tests/test_structured_output_live.py']))"
```

Result: **419 passed, 8 failed, 14 skipped.** The same 8 fail on `8272aea4` with the same command:
5 in `tests/release/test_linux_package.py` and 2 in `tests/release/test_windows_kernel_dll.py`
(they need a Linux release tree or a Windows kernel-DLL build tree), and
`tests/text/test_chat_templates.py::test_cpp_matches_independent_renderer` (it needs the built C++
renderer). The merge brings no new Python failure.
