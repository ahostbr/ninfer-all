#!/usr/bin/env python3
"""Writes a routing profile (--expert-profile) for a DeepSeek-V4.1 artifact whose experts were pruned
to a kept subset, from the built-in profile of the full release.

The built-in DeepSeek-V4.1 profile counts the release's 384 experts a block, so it cannot rank a
pruned pack's (the GSQ-RCO Coder keeps 192). The pack's kept-experts list says expert j of block L
is expert kept_experts[L][j] of the release; the converted artifact keeps that order, so the pack's
profile is the release's counts at the kept ids. The routes the pruned experts took go to others in
the pack, so these counts rank the pack's experts approximately: a profile only places experts (the
device cache, the --expert-host-mib tier) and never changes a result. A profile recorded on the pack
itself with --expert-profile-out ranks them exactly.

With --conversion (the artifact's .conversion.json) it also prints, for each byte budget, the share
of the profile's routes the experts that fit cover when ranked as the engine ranks them.

  python tools/expert_profile_kept.py --kept PACK.kept-experts.json \
      --artifact-id 7b52781e2d0a4a318fc629224e80f3ac --out pack.profile.json \
      [--conversion ARTIFACT.conversion.json --budgets-gib 30,40,50,60]
"""

import argparse
import json
import pathlib
import re

INC = pathlib.Path(__file__).resolve().parent.parent / "src/models/expert_profiles/builtin.inc"

# Bytes and values of one stored block, by artifact format name.
BLOCKS = {
    "gguf_iq1_s": (50, 256), "gguf_iq1_m": (56, 256), "gguf_iq2_xxs": (66, 256),
    "gguf_iq2_xs": (74, 256), "gguf_iq2_s": (82, 256), "gguf_iq3_xxs": (98, 256),
    "gguf_iq3_s": (110, 256), "gguf_iq4_xs": (136, 256), "gguf_iq4_nl": (18, 32),
    "gguf_q2_k": (84, 256), "gguf_q3_k": (110, 256), "gguf_q4_k": (144, 256),
    "gguf_q5_k": (176, 256), "gguf_q6_k": (210, 256), "gguf_q8_0": (34, 32),
    "gguf_mxfp4": (17, 32),
}


def builtin(family: str) -> tuple[list[int], list[int]]:
    """The built-in profile's bank widths and counts for `family`, as builtin.inc spells them."""
    text = INC.read_text()
    entry = re.search(r'\{"' + re.escape(family) + r'", (kWidths\d+), (kCounts\d+)\}', text)
    if not entry:
        raise SystemExit(f"{INC}: no built-in profile of {family}")

    def array(name: str) -> list[int]:
        body = re.search(r"\b" + name + r"\[\] = \{([^}]*)\}", text)
        return [int(v) for v in re.findall(r"\d+", body.group(1))]

    return array(entry.group(1)), array(entry.group(2))


def expert_bytes(conversion: dict, blocks: int, experts: int) -> list[int]:
    """Each block's bytes of one routed expert (its gate/up and down rows)."""
    sizes = [0] * blocks
    for method in conversion["methods"]:
        for parameter in method["parameters"]:
            found = re.fullmatch(r"text/layers/(\d+)/moe/(gate_up|down)", parameter)
            if not found:
                continue
            if method["format"] not in BLOCKS:
                raise SystemExit(f"{parameter}: unknown format {method['format']}")
            block_bytes, values = BLOCKS[method["format"]]
            rows, k = method["shape"]
            sizes[int(found.group(1))] += rows // experts * (k // values) * block_bytes
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kept", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-id", required=True, help="32 lowercase hex digits")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--family", default="deepseek-v4.1")
    parser.add_argument("--conversion", type=pathlib.Path)
    parser.add_argument("--budgets-gib", default="20,30,40,45,50,55,60")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.artifact_id):
        raise SystemExit("--artifact-id takes 32 lowercase hex digits")

    widths, counts = builtin(args.family)
    kept = json.loads(args.kept.read_text())
    blocks = kept["layers"]
    if len(widths) != blocks or any(w != kept["experts_per_layer"] for w in widths):
        raise SystemExit(f"the built-in profile's banks ({len(widths)} x {widths[0]}) are not the "
                         f"release the kept list prunes ({blocks} x {kept['experts_per_layer']})")
    profile = []
    for block in range(blocks):
        ids = kept["kept_experts"][str(block)]
        if len(ids) != kept["kept_per_layer"] or len(set(ids)) != len(ids):
            raise SystemExit(f"block {block}: the kept list is not {kept['kept_per_layer']} distinct ids")
        row = counts[block * widths[0]:(block + 1) * widths[0]]
        profile.append([row[i] for i in ids])
    args.out.write_text(json.dumps({"version": 1, "artifact_id": args.artifact_id,
                                    "counts": profile}) + "\n")
    print(f"wrote {args.out}: {blocks} blocks x {kept['kept_per_layer']} experts, "
          f"{sum(map(sum, profile))} of the release's {sum(counts)} routes")

    if args.conversion:
        sizes = expert_bytes(json.loads(args.conversion.read_text()), blocks, kept["kept_per_layer"])
        total_bytes = sum(sizes) * kept["kept_per_layer"]
        routes = sum(map(sum, profile))
        # The engine's order (models/deepseek_v41/model.cpp pin_hot_experts): count descending,
        # then expert id, then block; each expert taken while its bytes still fit.
        order = sorted(((-profile[b][e], e, b) for b in range(blocks)
                        for e in range(kept["kept_per_layer"])))
        print(f"experts: {total_bytes / 2**30:.1f} GiB in all")
        for gib in (float(v) for v in args.budgets_gib.split(",")):
            budget, used, covered = int(gib * 2**30), 0, 0
            for count, e, b in order:
                if used + sizes[b] <= budget:
                    used += sizes[b]
                    covered -= count
            print(f"  --expert-host-mib {int(gib * 1024)}: {used / 2**30:.1f} GiB pinned, "
                  f"{100 * used / total_bytes:.0f}% of the bytes, {100 * covered / routes:.1f}% "
                  f"of the profile's routes")


if __name__ == "__main__":
    main()
