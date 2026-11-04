# Building this fork natively on Windows (RTX 5090, sm_120a, CUDA 13.0.2)

**Does ninfer-all build natively on Windows? Yes.** Upstream carries its own native Windows port
(`docs/build-windows.md`, `scripts/build-native.bat`): file I/O on `CreateFileW` (`src/artifact/file_io.cpp`),
the 128-bit arithmetic as explicit limbs (`src/core/wide_math.h`), the NVFP4 TMA descriptors passed through
device memory on `WIN32` (`CMakeLists.txt:127-130`). None of the `ahostbr/ninfer-win` patches (bundle
`G:\REPO_CLONES\_bundles\ninfer-win-20261103.bundle`, based on `Neroued/ninfer@5b4303c0`) was needed; ninfer-all contains
5b4303c0 (`git cat-file -t 5b4303c0` is a commit, 1870 commits behind HEAD) and the port in it is a rewrite of the
same ideas. What this fork changed to build here is below, and it is small.

**Nothing was run on a GPU.** The build, `--help` / `--version` and the host tests are checked. Correctness and speed
on the 5090 are UNVERIFIED (section 5).

## 1. The build that was done

`docs/fork/build_windows.bat` (also at `G:\REPO_CLONES\ninfer-all-work\build.bat`), run in the background, logs in
`G:\REPO_CLONES\ninfer-all-work\{configure,build}.log`:

| | |
|---|---|
| host compiler | MSVC **19.44.35222**, toolset **14.44.35207** (VS 2022 Community), `vcvars64.bat -vcvars_ver=14.44`, `PROCESSOR_ARCHITECTURE=AMD64` |
| CUDA | **13.0.2** toolchain `C:\Strata-data\toolchain\cuda-13.0.2` (nvcc **13.0.88**, `CUDAToolkit_ROOT` and `CUDACXX` set, put first on `PATH` so the machine's 13.2 is never found) |
| arch | `-DCMAKE_CUDA_ARCHITECTURES=120a`, Release, Ninja (VS-bundled `ninja.exe`), `-DBUILD_TESTING=ON` |
| media | FFmpeg from the prebuilt vcpkg tree `-DNINFER_VCPKG_TREE=C:/vcpkg-work/installed/x64-windows` (avcodec-63, avformat-63, avutil-61, swscale-10, z); **vcpkg is not run**; no libcurl (N10) |
| build dir | `C:\Users\Ryan\ninfer-all-build144` (6.7 GB) |
| result | 1204 of 1204 steps, 0 failed; configure and build logs contain no download |
| size | `ninfer-serve.exe` 624,468,480 bytes (static CUDA runtime, sm_120a code) |
| time | configure 19:45:20, last link 20:08:02 on 2026-11-03: about 23 minutes at `-j12`, full build (the first, 14.38 build took longer because it ran beside other work and was built twice) |

`nvcc` 13.0.88 accepts MSVC 19.44 (`The CUDA compiler identification is NVIDIA 13.0.88 with host compiler MSVC 19.44.35222.0`).
The Visual Studio **2022 install's default toolset is 14.38**, and a plain `vcvars64.bat` takes it; see W4.

## 2. What was changed, and why

| ID | Where | Change | Why |
|---|---|---|---|
| W1 | `CMakeLists.txt` (the `NINFER_ARCH_BLACKWELL` guard) | new option `NINFER_ALLOW_CUDA_13_0` (default OFF) lets a 120a build use CUDA 13.0.x | upstream refuses 120a below 13.1: "12.8 and 12.9 miscompile sm_120a code" (its own finding for those two). Ryan's constraint is CUDA 13.0 because 13.2.51 miscompiles the sm_120 IQ kernels. 13.0 is neither of the two upstream names nor in the 13.1+ it qualified, so **this is the one place the build leaves upstream's qualified set**. The build script passes it ON |
| W2 | `small_t_fp8.cu:50`, `small_t_k8v4.cu:48`, `small_t_nvfp4.cu:45` (`DynamicBytes`), `small_t_i8_launch.cuh:97` (`kDynamicBytes`), `small_t.cu:517` (`kWidest`) | `constexpr` becomes `static constexpr` | MSVC `C3495` ("a simple capture must be a variable with automatic storage duration declared in the reaching scope of the lambda") on a local `constexpr` used inside a lambda nested in a generic lambda. 17 CUDA translation units failed in the first pass (`small_t.cu`, `small_t_fp8.cu`, `small_t_k8v4.cu`, `small_t_nvfp4.cu` and 13 of the `small_t_i8_w*` units that include the shared header). A `static` one is not captured. Values and generated code do not change |
| W3 | `tests/models/qwen4_exp/test_engine_real.cpp:539` | `const auto* top` becomes `const auto top` | `std::array::begin()` is a class under MSVC, not a pointer (`C3535`). Test code only |
| W4 | `CMakeLists.txt`, `docs/fork/msvc_19_38_wide_add.cpp` | refuse MSVC older than 19.40 | **cl 19.38.33145 at /O2 miscompiles `core::wide_add`** (`src/core/wide_math.h:42`): with constant inputs `(2^128-1)+1` reports no overflow. The first full build used it by default and `ninfer_wide_math_test` failed on exactly that (2 checks). The header is correct: `/Od`, volatile inputs and cl 19.44.35222 at /O1 and /O2 all give the right answer (the repro prints both). The build was redone from scratch with 14.44 |

Not changed: any kernel body, any numeric path, any file under `src/` apart from the network cuts and the four constexpr
spellings above. `docs/build-windows.md` still names CUDA 12.8 and VS 2022 BuildTools; it is upstream prose.

## 3. Checks

| check | result |
|---|---|
| `ninfer-serve.exe --version` | `ninfer-serve 2026-11-02.1+2026-11-02.1-29-g4bfb02a4` (the stamp is the commit it was built from) |
| `ninfer-serve.exe --help` | prints; every flag LiteSuite and LiteTUI pass is in it: `--host --port --model-id --max-context --max-concurrency --spec --draft-tokens --lm-head-draft --kv-dtype --kv-capacity --host-kv-mib --preserve-thinking --vision`; and the DeepSeek ones `--expert-residency --ngram-table --ngram-residency`; `--spec` takes `none|mtp|dflash|dflash2|dspark` |
| `ninfer-serve.exe MODEL --host 0.0.0.0` | refuses before touching the model: `--host must be 127.0.0.1, localhost or ::1: this build listens on this PC only` |
| `ctest` with `CUDA_VISIBLE_DEVICES=-1` (no GPU visible, so no GPU use) | **370 tests: 103 passed, 266 skipped (they need a device, exit 77), 1 failed** |
| the failed one | `ninfer_multi_gpu_test`: `create_directories ... "bus/pci/devices\0000:01:00.0"`, its fixture of a fake Linux sysfs names a directory with `:`, which NTFS refuses. Test portability, not the product (`src/core/peer_preflight.cpp:61` opens Linux sysfs files with `ifstream`; on Windows they do not exist and the facts stay empty) |
| tests of the cuts | `ninfer_mcp_proxy_test`, `ninfer_serve_options_test` (the new loopback refusals), `ninfer_prompt_input_test`, `ninfer_http_routes_test`, `ninfer_wide_math_test`: passed |
| Python | `tests/convert/test_deepseek_v41.py` 21 passed, `test_official_recipes.py` 25 passed, `tests/release/test_universal_package.py` 6 passed. `test_linux_package.py` (5 tests) and `test_windows_kernel_dll.py` (2) cannot run on this box: they fake `nvcc` with a shell script, or call `cmake -G Ninja` with no Ninja on `PATH`. They are Linux-packager tests and would fail on Windows without this fork |
| network scanner | `python -I docs/fork/audit_network.py` exit 0 |

## 4. Staged engine

`C:\Users\Ryan\.litesuite\llm\ninfer-all-<shortsha>\`: `ninfer-serve.exe`, `ninfer.exe`, the five FFmpeg DLLs, `VERSION`,
`SHA256SUMS`, `FORK-INFO.txt`. It needs the Microsoft Visual C++ 2015-2022 x64 Redistributable like the live one. The live
`C:\Users\Ryan\.litesuite\llm\ninfer` (v0.1.0-win, `ahostbr/ninfer-win@40f207f2`) and `ninfer.prev` were not touched. The live
folder has a `libcurl.dll`; this build has none.

How the tools find an engine: LiteTUI uses `settings.ninfer_executable`, env `LITETUI_NINFER_EXE`, else
`~/.litesuite/llm/ninfer/ninfer-serve.exe` (`ninfer_engine.py:104-106`, `settings.py:700`) and builds its argv at
`ninfer_engine.py:210-241`. LiteSuite installs into `~/.litesuite/llm/ninfer` by swapping with `ninfer.prev`
(`ninfer-install.ts:391-394`); it has no executable override, and its installer downloads
`ahostbr/ninfer-win` releases (`ninfer-install.ts` `NINFER_REPO`), a repository that is private now.

## 5. Not proven

1. **No GPU run.** Not one kernel of this build has executed. CUDA 13.0.2 on sm_120a is outside what upstream qualified (W1);
   upstream's finding is that older toolkits produced wrong kernels. The first thing to do with Ryan's go is the 0.8B model
   (`qwen3_5_0_8b.ninfer`) against a known answer, then the GPU part of the test suite.
2. **Pinned memory under WDDM.** Upstream says WDDM charges pinned host memory against the card (`docs/build-windows.md:87`);
   `--expert-residency host` pins 61.5 GiB for the Coder. Nobody has measured that here (CODER.md section d).
3. **`NINFER_DIRECTSTORAGE`** is refused (N01), so `--disk-kv-directstorage` is not offered.
4. The Qwen artifact check is by reading code and the artifact header, not by opening it with the engine (`docs/fork/QWEN_ARTIFACT.md`).
