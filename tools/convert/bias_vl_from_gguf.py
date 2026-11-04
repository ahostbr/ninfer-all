"""The routing bias of image positions, from a GGUF that carries it, as the safetensors file the
`vision` component's `--source bias_vl=FILE` reads.

ISTA-DASLab's Coder GGUF stores the release's `layers.N.ffn.gate.bias_vl` as `blk.N.exp_probs_b_vl.bias`
(one F32 vector per block, over the experts the GGUF keeps), so the vision tower converts without
reading the base release's shards:

    python -m tools.convert.bias_vl_from_gguf SHARD1.gguf --out bias_vl.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .deepseek_v41 import BIAS_VL, GGUFShards
from .fetch_tensors import write_safetensors


def bias_vl_tensors(gguf: GGUFShards) -> dict[str, tuple[dict, bytes]]:
    layers = int(gguf.kv["deepseek41.block_count"])
    out = {}
    for layer in range(layers):
        name = f"blk.{layer}.exp_probs_b_vl.bias"
        info = gguf.info(name)
        if info.type_name != "F32" or len(info.shape) != 1:
            raise ValueError(f"{name}: expected an F32 vector, found {info.type_name} {info.shape}")
        values = np.ascontiguousarray(gguf.file(name).read_direct(name).astype("<f4"))
        out[BIAS_VL.format(layer)] = ({"dtype": "F32", "shape": [int(info.shape[0])]},
                                      values.tobytes())
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gguf", type=Path, help="the first shard of the split GGUF")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with GGUFShards(args.gguf) as gguf:
        tensors = bias_vl_tensors(gguf)
    write_safetensors(args.out, tensors, {"source": str(args.gguf)})
    print(f"wrote {args.out}: {len(tensors)} tensors", flush=True)


if __name__ == "__main__":
    main()
