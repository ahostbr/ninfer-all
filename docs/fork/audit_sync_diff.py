#!/usr/bin/env python
"""docs/fork/audit_sync_diff.py - scan the lines an upstream sync ADDED for network, download and
process constructs.

    python -I docs/fork/audit_sync_diff.py <fork-parent> [<merge>]

audit_network.py judges files: a new hit inside an already-allowed file passes it. This script
reads `git diff -U0 <fork-parent> <merge>` (docs/ and third_party/ excluded) and prints every added
line that matches audit_network.py's patterns, plus process launches, telemetry words and URLs, so
each one can be judged by a person. It reads git output only and opens no connection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit_network", HERE / "audit_network.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

EXTRA = re.compile(r"\bsystem\(|popen\(|CreateProcess|ShellExecute|posix_spawn|subprocess\.|os\.system"
                   r"|telemetry|analytics|sentry|posthog|https?://")
PATTERNS = (audit.SCRIPT_PAT, audit.CODE_PAT, EXTRA)


def main() -> int:
    sys.stdout.reconfigure(errors="backslashreplace")
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    base, head = sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "HEAD"
    diff = subprocess.run(["git", "diff", "-U0", base, head, "--", ".", ":!docs", ":!third_party"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=True).stdout
    current, hits = None, {}
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            if any(p.search(line[1:]) for p in PATTERNS):
                hits.setdefault(current, []).append(line[1:].strip()[:160])
    for name, lines in sorted(hits.items()):
        print(f"== {name} ({len(lines)})")
        for text in lines:
            print("   ", text)
    print(f"audit_sync_diff: {sum(map(len, hits.values()))} added lines in {len(hits)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
