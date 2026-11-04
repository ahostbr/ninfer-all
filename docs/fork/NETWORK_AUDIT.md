# Network audit of ninfer-all c37a053b and what this fork did about each path

Base: `iamwavecut/ninfer-all` `master` at `c37a053b9cec652991c5835cbf006591b454f86e` (merge of PR #211,
2026-11-04 by commit date). Line numbers in the "Where" column are the base's unless a column says
"fork". The audit was done by reading and grepping the tree, with the method of the Strata fork's
`NETWORK_AUDIT.md` (`F:/REPO_CLONES/Strata-fork/fork/NETWORK_AUDIT.md`), adapted to this repository.

Status: static analysis, plus a build whose configure and build logs contain no download
(`grep -iE "download|https?://|git clone"` on both is empty). **UNVERIFIED by Ryan.** The dynamic
proofs (a build with the network blocked, the running server with no outbound connection) are not
done; section 6 says exactly what is missing.

This fork is local. Nothing in it is pushed. Each cut is its own commit where the files allow it, so a
cut is un-done with `git revert <sha>` (section 7).

## 1. How to re-check (paste and run)

From the root of the tree under test. Nothing here opens a connection.

```bash
# the scanner: exit 0 = every hit is in a file this audit judged (the judgement is the ALLOWED table in the script)
python -I docs/fork/audit_network.py            # add -v to list the allowed hits (180 on this HEAD)

# the raw greps behind it (expected output on this HEAD in brackets)
grep -rnE "FetchContent|ExternalProject|file\(DOWNLOAD|file\(UPLOAD|git clone" --include=CMakeLists.txt --include=*.cmake . | grep -vE "^\./(docs|bench)/" | grep -vE "^[^:]+:[0-9]+:\s*#"
#   [CMakeLists.txt: the message that names FetchContent in the FATAL_ERROR; cmake/NinferFFmpeg.cmake: include(ExternalProject) and two ExternalProject_Add over LOCAL archives (Linux only)]
grep -rnE "curl/curl.h|curl_easy|getaddrinfo|WSAStartup|httplib::Client" src apps include --include=*.cpp --include=*.h --include=*.cu
#   [src/serve/mcp_proxy.cpp and .h: httplib::Client, target restricted to loopback (N11)]
grep -rnE "\bsystem\(|popen\(|CreateProcess|ShellExecute|posix_spawn" src apps include --include=*.cpp --include=*.h --include=*.cu
#   [no output]
grep -rnE "bind_to_port|listen_after_bind" src --include=*.cpp
#   [src/serve/http_server.cpp: the one listener, plus the optional stats port]
grep -rnE "tcp_sink|udp_sink|syslog_sink" src apps tests --include=*.cpp --include=*.h
#   [no output: spdlog ships network sinks as headers, nothing includes them]
grep -rnE "OPENSSL|CPPHTTPLIB_" --include=CMakeLists.txt --include=*.cmake .
#   [no output: cpp-httplib is built without TLS]

# the tests of the cuts
python -m pytest tests/convert/test_deepseek_v41.py -q          # [21 passed]
# C++: tests/test_mcp_proxy.cpp and tests/test_serve_options.cpp (built by the Windows build; CPU only)
```

## 2. Every path found, and what was done

Dispositions: **removed** (code gone), **refused** (code kept, fails closed), **kept** (section 4 says why).
"Commit" is the one that made the change (filled in section 8).

### 2a. Build time

| ID | Where (base c37a053b) | What it did | Disposition |
|---|---|---|---|
| N01 | `CMakeLists.txt:175-179` | with `-DNINFER_DIRECTSTORAGE=ON` (Windows, off by default): `FetchContent` of `www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0` at configure time | **refused**: the block is a `FATAL_ERROR` that says so |
| N02 | `cmake/FetchMediaSources.cmake:25` (`file(DOWNLOAD ... TLS_VERIFY ON)`), `scripts/fetch-media-sources.sh`, callers `scripts/build.sh:98`, `scripts/build-native.sh:59` | downloaded the FFmpeg and dav1d source archives from `ffmpeg.org` and `downloads.videolan.org` for the Linux build | **removed**; the callers no longer fetch; `cmake/NinferFFmpeg.cmake` now says to place the archives by hand. Its two `ExternalProject_Add` (`:69`, `:85`) read LOCAL archives with a pinned SHA-256 and run dav1d's meson with `--wrap-mode=nodownload` (`:25`): kept, Linux only |
| N10 | `src/product/media_acquire/acquire.cpp:3,166,267,356`; `cmake/Dependencies.cmake:62`; `src/product/CMakeLists.txt:5`; `vcpkg.json:7`; `cmake/NinferTargets.cmake:108` (`libcurl*.dll`) | libcurl. A request that named an `http(s)` image or video URL made the server resolve it (`getaddrinfo`) and fetch it (`curl_easy_perform`), with a private-address filter. `prompt_input.cpp:80` turned the filter off for the CLI | **removed**: the URL branch throws `invalid_argument` ("fetching a media URL is disabled in this build; send the media as a data: URI"), which the API returns as 400 `invalid_media`. `data:` URIs, local paths and byte buffers still work. libcurl is out of CMake, `vcpkg.json` and the DLL staging |
| N11 | `src/serve/mcp_proxy.cpp:39,144`, `src/serve/http_server.cpp:746-752`, `serve_options.cpp:1597` | with `--webui-mcp-proxy` (off by default; `tests/test_serve_options.cpp:591` asserts that): a relay at `/cors-proxy` that `httplib::Client`-connected to any `http://host:port` the browser page named | **refused**: `parse_mcp_proxy_target` accepts `127.0.0.1`, `localhost` and `::1` only; tests added to `tests/test_mcp_proxy.cpp` |
| N03 | `scripts/package-release.ps1`, `scripts/package-linux-release.py:234,239` | packaging copied the removed `download-model.*` and `docker/` files | **edited** to stop copying them (release packaging is not used here) |

### 2b. The running server

| ID | Where | What it did | Disposition |
|---|---|---|---|
| N30 | `src/serve/serve_options.cpp:675` (`--host`), `src/serve/http_server.cpp:1477-1479` (`bind_to_port(options_.host, port)`, and the stats port), default `127.0.0.1` at `serve_options.h:34` | `--host 0.0.0.0` or any address exposed the API (and `--stats-port`) beyond this PC | **refused**: `--host` accepts `127.0.0.1`, `localhost` and `::1` only; the help text says so; tests in `tests/test_serve_options.cpp` |

### 2c. Install, update and download scripts

| ID | Where | What it did | Disposition |
|---|---|---|---|
| N19 | `Dockerfile:119` (`NINFER_HOST=0.0.0.0`), `compose.yaml`, `docker/`, `flake.nix:28` (`fetchurl`), `flake.lock`, `tests/test_docker_build_cache.sh` | container and nix packaging: `apt`, `pip`, image pulls, a server bound to all interfaces | **removed** |
| N40 | `scripts/download-model.bat:148,158` (`curl.exe -L -C -` from `huggingface.co`), `scripts/download-model.sh:32`, `scripts/standalone-windows/install.ps1:176,250` (`api.github.com`, `Invoke-Download`), `scripts/fetch-webui.sh:18` (`curl` from `github.com/ggml-org/llama.cpp`), `scripts/check-linux-scripts.sh` (stubs of the above) | model, release and WebUI downloaders | **removed** |
| N41 | `.github/workflows/*.yml` (9), `.github/scripts/` (HF upload of build caches, CUDA licence fetch, compiler-cache install, release publishing), `scripts/pods/` (170 files: rented-host bootstrap with `git clone`, `apt`, `pip`, Hugging Face upload and download), `scripts/multi-gpu-testing/onstart-pipeline.sh:49`, `scripts/sm80-probe/probe.sh:186` (`git clone`) | CI and the maintainer's rented-GPU harness | **removed** with their tests (`tests/pods/`, 12 files of `tests/release/` that import `.github`, `tests/reference/test_fetch_slice.py`). The five `tests/release/` files that test local packaging stay |
| N20 | `tools/convert/fetch_tensors.py:28,46-54,115-117` | `ENDPOINT = "https://huggingface.co"`, ranged `GET`s of the base release's safetensors index, headers and tensors, `HF_TOKEN` | **cut to local files**: the hub reader is gone; `main` reads an index and shards from a directory; `fetch` and `write_safetensors` (used by the tests and `bias_vl_from_gguf.py`) are unchanged |
| N21 | `tools/eval/build_heldout_corpus.py:105` (urllib crawl of arXiv, Wikipedia, the GitHub API), `tools/reference/fetch_slice.py:37` (HF range requests) | corpus and checkpoint-slice downloaders | **removed** |

## 3. What the Windows build does and does not do on the network

- `scripts/build-native.bat` and `docs/build-windows.md` expect a prebuilt vcpkg tree. This fork's
  offline build passes `-DNINFER_VCPKG_TREE=C:/vcpkg-work/installed/x64-windows` and **does not run
  vcpkg**. Configuring with the vcpkg toolchain instead would run `vcpkg install` in manifest mode
  (`vcpkg.json`, `builtin-baseline`), which downloads ports. Do not.
- The only thing the compiler chain itself may send is Microsoft's `vctip.exe` telemetry
  (Visual Studio / MSVC, not this repository). The build `.bat` sets `VSCMD_SKIP_SENDTELEMETRY=1`.
  `vctip.exe` is not blocked by that variable (the Strata audit's `MSVC_BIN_FIRST` trick is the
  workaround). Not done here.

## 4. Kept, and why

| ID | Where | What it is | Why it is kept |
|---|---|---|---|
| K1 | `src/serve/http_server.cpp:1477-1479` | **the only listener.** cpp-httplib, default `127.0.0.1`, `--host` loopback-only (N30) | the server is the product |
| K2 | `tools/monitor/monitor.py:1133,1165` | dashboard `ThreadingHTTPServer((cfg.bind, cfg.port))`, `--bind` defaults to `127.0.0.1`; polls the local server's `/stats` | optional tool; `--bind` can still be set to another address by hand |
| K3 | `tools/streaming_http/client.py`, `tools/ninfer_serve/client.py`, `tools/bench/*.py`, `bench/**/*.py`, `tools/smoke/*.py`, `examples/cli/send_to_serve.py` | clients of a server the user starts, on `127.0.0.1` | benchmarks and probes; the host is an argument |
| K4 | `src/serve/http_transport.cpp:7-9,205-208` | `setsockopt` (keep-alive, user timeout) on the accepted server socket | no new connection |
| K5 | `scripts/build.sh:78`, `cmake/FindFFMPEG.cmake:32`, `tools/convert/sources/gguf_source.py:12,68`, `tools/generate_jinja_unicode.py:4` | printed advice (`apt install`, `vcpkg install`, `pip install gguf`) and a docstring URL | nothing runs them |
| K6 | `eval/*.sh` | `curl` health checks of the server they start on `127.0.0.1:18080` | local. **`eval/ninfer_eval` drives the `evalscope` package, which downloads datasets from ModelScope / Hugging Face when it is installed and run** (`eval/requirements.txt`; `backend_args.dataset_hub`, `allow_network_downloads` at `eval/ninfer_eval/backends/evalscope.py:99,148`). It is not installed here. Treat running it as a download |
| K7 | `third_party/cpp-httplib/httplib.h`, `third_party/spdlog/.../tcp_client*.h udp_client*.h` | socket code of the vendored libraries | httplib is the listener's library and is built without TLS; the spdlog network sinks are headers nothing includes |
| K8 | `tools/bench/make_bench_corpus.py:166`, `tools/bench/ttft/build_fixtures.py:54` | `AutoTokenizer.from_pretrained(path, local_files_only=True)` | local only by the argument |
| K9 | `tools/reference/gemma4_vision.py:135` | `AutoConfig.from_pretrained(hf_dir)` | a local directory |
| K10 | `eval/corpora/**` | text corpora that contain URLs as text | data, never executed |

## 5. Not proven, or open

1. **No dynamic proof.** Section 6.
2. **The scanner is static and per file.** An allowance covers every hit in that file. A new network
   call added to an allowed file (say `src/serve/http_server.cpp`) would not be caught; the greps in
   section 1 are the second line of defence.
3. **Docs are stale.** `docs/` still describes `download-model`, the Docker image, `fetch-webui`, the
   nix flake and the CI. They are upstream prose; editing them would conflict with every rebase.
   `docs/fork/WINDOWS_PORT.md` is the replacement for the Windows build steps.
4. **`eval/ninfer_eval` and the Python tools' optional packages** (`transformers`, `evalscope`,
   `huggingface_hub` in `scripts/`) are not installed by anything here; if someone installs them, what
   they do is outside this audit.
5. **LiteSuite's own installer** (`apps/desktop/src/litesuite/services/llm/ninfer-install.ts`,
   `NINFER_REPO = "ahostbr/ninfer-win"`, `NINFER_RELEASES_API`) downloads from GitHub. That is
   LiteSuite's code, not this repository's, and the repository it names is private now.

## 6. What the dynamic proofs would be (not done)

- **Build with the network blocked.** Run `build.bat` under an AppContainer or a firewall rule that
  denies outbound traffic from `cmake.exe`, `ninja.exe`, `cl.exe`, `nvcc.exe` and `link.exe`, and show
  a census of their sockets as empty (the Strata fork's `fork/offline_build_proof.py` is the model).
- **Server with no outbound connection.** Start `ninfer-serve` on a model and list its TCP table
  (`Get-NetTCPConnection -OwningProcess <pid>`): only the listener on `127.0.0.1`. That needs a model
  load, which needs Ryan's go.

## 7. Reverting one cut

Each cut is in the commit named in section 8; `git revert <sha>` restores it. N01 and N02 restore a
configure-time download, N10 and N11 restore a server that connects out, N30 restores `--host 0.0.0.0`.
Do not.

## 8. Commits

All on the local branch `fork/audited`, based on `c37a053b`. Nothing is pushed.

| commit | what | IDs |
|---|---|---|
| `f0e0280a` | cut(net): no configure-time or script downloads, no CI, container, nix or rented-host harness | N01 N02 N03 N19 N20 N21 N40 N41 |
| `11551892` | feat(convert): the Coder GGUF dialect; `bias_vl_from_gguf.py` (replaces the network read of the vision bias) | (CODER.md) |
| `bcc07f9f` | cut(net): no HTTP client, loopback-only listener and MCP relay | N10 N11 N30 |
| `337e8ec2` | fix(windows): MSVC C3495, the CUDA 13.0 escape hatch | W1 W2 W3 |
| `4bfb02a4` | fix(windows): refuse MSVC 19.38 | W4 |
