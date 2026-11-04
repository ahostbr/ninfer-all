#!/usr/bin/env python
"""docs/fork/audit_network.py - the re-runnable half of docs/fork/NETWORK_AUDIT.md.

    python -I docs/fork/audit_network.py [--root DIR] [-v]

Scans the tree for the constructs that open a connection, fetch a file, install something or start a
listener, and fails (exit 1) when one appears in a file the audit has not judged: ALLOWED below is
that judgement, one line of reasoning per file. A new hit in an allowed file is not caught (the
allowance is per file); a hit in any other file is. It reads files only and opens no connection.

Method adapted from the Strata fork's fork/audit_network.py (F:/REPO_CLONES/Strata-fork).
docs/ (prose), third_party/ (vendored libraries, judged separately in the audit) and build trees
are not scanned. Exit 0 = every hit is in an allowed file.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "build", "build-ninja", "build-windows", "__pycache__", "docs", "node_modules"}
SKIP_PREFIX = ("third_party/",)
SCRIPT_SUFFIX = {".py", ".sh", ".bat", ".cmd", ".ps1", ".cmake", ".yml", ".yaml", ".js", ".mjs",
                 ".html", ".nix", ".in", ".toml", ".json"}
SCRIPT_NAMES = {"CMakeLists.txt", "Dockerfile"}
CODE_SUFFIX = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc"}

SCRIPT_PAT = re.compile(
    r"urllib\.request|urlopen|urlretrieve|http\.client|\bimport requests\b|requests\.(get|post|put|head|Session)"
    r"|httpx|aiohttp|\bsocket\.|ThreadingHTTPServer|HTTPServer\(|webbrowser|\bcurl\b|\bwget\b|\baria2c?\b"
    r"|Invoke-WebRequest|Invoke-RestMethod|Start-BitsTransfer|git (clone|pull|fetch|submodule)"
    r"|pip3? (install|download)|-m pip\b|apt(-get)? install|docker (pull|push|login)|npm (install|ci)"
    r"|FetchContent|ExternalProject|file\(DOWNLOAD|file\(UPLOAD|huggingface\.co|hf\.co/|api\.github\.com"
    r"|releases/(latest|download)|from_pretrained|hf_hub_download|snapshot_download"
    r"|huggingface_hub|\bhf (download|upload)|HF_TOKEN|HF_ENDPOINT|nuget\.org|vcpkg install")
CODE_PAT = re.compile(
    r"curl/curl\.h|curl_easy|winsock|WSAStartup|getaddrinfo|gethostbyname|InternetOpen|WinHttp|AF_INET"
    r"|sys/socket\.h|arpa/inet\.h|httplib::Client|bind_to_port|listen_after_bind|CPPHTTPLIB_OPENSSL")

# file -> why a hit there is not an outbound path. Keep this list SHORT: every line is a claim
# docs/fork/NETWORK_AUDIT.md defends.
ALLOWED: dict[str, str] = {
    "docs/fork/audit_network.py": "this scanner: its patterns are the hits",
    "src/serve/http_server.cpp": "K1: the one listener (cpp-httplib bind_to_port); --host is loopback-only since N30",
    "src/serve/mcp_proxy.cpp": "N11: httplib::Client to a target that parse_mcp_proxy_target restricts to loopback; the relay itself is off by default",
    "src/serve/mcp_proxy.h": "N11: declares the relay",
    "tools/monitor/monitor.py": "K2: dashboard; binds --bind (default 127.0.0.1), polls the local server's /stats",
    "tools/monitor/watchdog.py": "K2: polls the local server's /stats",
    "tools/streaming_http/client.py": "K3: HTTP client of the local server used by the tools",
    "tools/ninfer_serve/client.py": "K3: client of the local server",
    "tools/longctx_recall_probe.py": "K3: client of the server --url names (default local)",
    "tools/smoke/prefix_reuse_issues.py": "K3: loopback probe",
    "tools/smoke/serve_contract.py": "K3: client of the local server",
    "tools/smoke/serve_thinking_preservation.py": "K3: loopback server it starts",
    "tools/smoke/vision_stall_probe.py": "K3: client of the local server",
    "tools/smoke/overlay_ab.py": "K3: client of the local server",
    "tools/bench/refbench.py": "K3: client of the server --url names",
    "tools/bench/ttft/report.py": "K3: `requests` is a dict named requests; no HTTP",
    "tests/test_structured_output_live.py": "K4: live test against a server it starts on 127.0.0.1",
    "examples/cli/send_to_serve.py": "K3: example client of the local server",
    "tools/bench/make_bench_corpus.py": "N21: AutoTokenizer.from_pretrained with local_files_only (see audit)",
    "tools/bench/ttft/build_fixtures.py": "N21: AutoTokenizer.from_pretrained with local_files_only (see audit)",
    "tools/reference/gemma4_vision.py": "N21: AutoConfig.from_pretrained of a LOCAL directory",
    "CMakeLists.txt": "N01: FetchContent removed; the NINFER_DIRECTSTORAGE block is a FATAL_ERROR",
    "cmake/Dependencies.cmake": "N10: libcurl removed",
    "cmake/NinferFFmpeg.cmake": "N02: ExternalProject over LOCAL archives (URL is a file path, URL_HASH pinned); meson runs --wrap-mode=nodownload; Linux only",
    "cmake/MediaSources.cmake": "N02: pinned digests and the upstream URL text; nothing in the build opens the URL",
    "src/product/media_acquire/acquire.cpp": "N10: the URL branch throws; no socket code remains",
    "src/serve/http_transport.cpp": "K1: setsockopt on the accepted server socket (keep-alive, user timeout); opens no connection",
    "tools/convert/fetch_tensors.py": "N20: the docstring names the removed behaviour; the code reads local files only",
    "tools/convert/sources/gguf_source.py": "K5: a message that names `pip install gguf`; nothing is run",
    "tools/generate_jinja_unicode.py": "K5: reads a local directory of Unicode data files; the docstring names where a person gets them",
    "scripts/build.sh": "K5: a message that names apt packages; nothing is run",
    "cmake/FindFFMPEG.cmake": "K5: a message that names `vcpkg install`; nothing is run",
    "eval/run_qwen3_8_27b_nvfp4_reasoning.sh": "K6: curl health check of the server it starts on 127.0.0.1:18080",
    "eval/run_qwen3_8_27b_nvfp4_gpqa_budgets.sh": "K6: same",
    "eval/run_qwen3_8_27b_gsq_rco_reasoning.sh": "K6: same",
    "eval/run_qwen3_8_27b_groupwise_reasoning.sh": "K6: same",
}
ALLOWED_PREFIX: dict[str, str] = {
    "tests/": "tests: loopback servers, mocks and placeholder hosts (the URL rule below still applies)",
    "bench/": "bench runners: clients of a server they start on 127.0.0.1",
    "tools/bench/": "bench runners: clients of a server on 127.0.0.1",
}

URL_PAT = re.compile(r"https?://(\[[0-9a-fA-F:]+\]|[A-Za-z0-9._-]+)(?::\d+)?")
URL_LOCAL = re.compile(
    r"^(127\.\d+\.\d+\.\d+|localhost|\[?::1\]?|0\.0\.0\.0|x|host|.*example(\.[a-z]+)?|.*\.invalid"
    r"|.*\.test|.*\.localhost|host:port|www\.w3\.org)(:\d+)?$")
URL_OK: dict[str, str] = {
    "docs/fork/audit_network.py": "this scanner",
    "cmake/MediaSources.cmake": "N02: the upstream release URLs as text next to the pinned digests",
    "LICENSE": "licence text",
    "README.md": "prose",
    "CONTRIBUTING.md": "prose",
    "CONTRIBUTORS.md": "prose",
    "AGENTS.md": "prose",
    "vcpkg.json": "N10: the $schema URL of the manifest",
    "tools/generate_jinja_unicode.py": "K5: docstring names where the Unicode data files come from; the script reads a local directory",
}
URL_OK_PREFIX: dict[str, str] = {
    "src/": "C++ attribution comments and strings",
    "include/": "C++ attribution comments",
    "tests/": "tests: placeholder and mock hosts",
    "tools/chat_templates/": "chat-template attribution",
    "tools/convert/templates/": "chat-template attribution",
    "tools/convert/": "attribution comments in the converters",
    "bench/": "bench data and reports (text)",
    "eval/corpora/": "perplexity corpora: documents that contain URLs as text, never executed",
}

COMMENT = ("#", "//", "rem ", "::", "*", "/*")
CODE_COMMENT = ("//", "*", "/*")


def skipped(rel: str) -> bool:
    parts = rel.split("/")
    return any(part in SKIP_DIRS for part in parts[:-1]) or rel.startswith(SKIP_PREFIX)


def allowed(rel: str):
    if rel in ALLOWED:
        return ALLOWED[rel]
    return next((why for p, why in ALLOWED_PREFIX.items() if rel.startswith(p)), None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent.parent))
    ap.add_argument("-v", action="store_true", help="also print the hits in allowed files")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    bad = seen = files = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if skipped(rel) and rel != "docs/fork/audit_network.py":
            continue
        if p.suffix in CODE_SUFFIX:
            pat, skip = CODE_PAT, CODE_COMMENT
        elif p.suffix in SCRIPT_SUFFIX or p.name in SCRIPT_NAMES:
            pat, skip = SCRIPT_PAT, COMMENT
        else:
            continue
        files += 1
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        why = allowed(rel)
        url_why = URL_OK.get(rel) or next((w for pre, w in URL_OK_PREFIX.items() if rel.startswith(pre)), None)
        for n, line in enumerate(lines, 1):
            stripped = line.lstrip().lower()
            if stripped.startswith(skip) and p.suffix not in {".json"}:
                continue
            if url_why is None:
                for m in URL_PAT.finditer(line):
                    host = m.group(1).rstrip(".:")
                    if not URL_LOCAL.match(host):
                        bad += 1
                        print(f"UNEXPECTED {rel}:{n}: URL to {host}: {line.strip()[:120]}")
            if pat.search(line):
                seen += 1
                if why is None:
                    bad += 1
                    print(f"UNEXPECTED {rel}:{n}: {line.strip()[:140]}")
                elif a.v:
                    print(f"allowed    {rel}:{n}: {line.strip()[:100]}")
    print(f"audit_network: {files} files scanned, {seen} hits, {bad} unexpected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
