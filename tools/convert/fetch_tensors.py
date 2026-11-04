"""Named tensors of a safetensors release, read by byte range from LOCAL files into one file.

Fork cut N20 (docs/fork/NETWORK_AUDIT.md): upstream read these tensors from huggingface.co (the
index, each shard's header and the matching bytes, by HTTP range requests, with HF_TOKEN). This build
opens no connection: `main` reads a directory that already holds the release's index and shards.

A converter source can need a few small tensors of a release whose shards it does not otherwise
read, such as DeepSeek-V4.1's routing bias of image positions, one vector per block spread over
43 of its 48 shards. This reads the index, each needed shard's header and the matching tensors'
bytes, nothing else, and writes them, as stored, into one safetensors file:

    python -m tools.convert.fetch_tensors RELEASE_DIR \\
        --match 'layers.*.ffn.gate.bias_vl' --out models/deepseek-v4.1/bias_vl.safetensors

`--match` takes fnmatch patterns over the tensor names (repeatable). The Coder GGUF carries the same
bias as `blk.N.exp_probs_b_vl.bias`; tools/convert/bias_vl_from_gguf.py writes the file from that.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import struct
from typing import Callable

INDEX = "model.safetensors.index.json"

# read(file, begin, end) -> the file's bytes [begin, end)
Reader = Callable[[str, int, int], bytes]


def local_reader(directory: Path) -> Reader:
    def read(file: str, begin: int, end: int) -> bytes:
        with open(directory / file, "rb") as stream:
            stream.seek(begin)
            data = stream.read(end - begin)
        if len(data) != end - begin:
            raise OSError(f"{file}: asked for bytes {begin}..{end}, got {len(data)} bytes")
        return data

    return read


def _header(read: Reader, file: str) -> tuple[int, dict]:
    size = struct.unpack("<Q", read(file, 0, 8))[0]
    return 8 + size, json.loads(read(file, 8, 8 + size))


def fetch(read: Reader, index: dict, patterns: list[str]) -> dict[str, tuple[dict, bytes]]:
    """The tensors whose names match a pattern: name -> (header entry, bytes), in name order."""

    names = sorted(name for name in index["weight_map"]
                   if any(fnmatch.fnmatchcase(name, p) for p in patterns))
    if not names:
        raise ValueError(f"no tensor of the index matches {patterns}")
    headers: dict[str, tuple[int, dict]] = {}
    out = {}
    for name in names:
        file = index["weight_map"][name]
        if file not in headers:
            headers[file] = _header(read, file)
        base, header = headers[file]
        entry = header[name]
        begin, end = entry["data_offsets"]
        out[name] = (entry, read(file, base + begin, base + end))
    return out


def write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]],
                      metadata: dict[str, str]) -> None:
    header, offset = {"__metadata__": metadata}, 0
    for name, (entry, data) in tensors.items():
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for _, data in tensors.values():
            out.write(data)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("release_dir", type=Path, help="a directory with the index and the shards")
    parser.add_argument("--match", action="append", required=True, metavar="PATTERN")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    index = json.loads((args.release_dir / INDEX).read_text())
    tensors = fetch(local_reader(args.release_dir), index, args.match)
    write_safetensors(args.out, tensors, {"source": str(args.release_dir)})
    total = sum(len(data) for _, data in tensors.values())
    print(f"wrote {args.out}: {len(tensors)} tensors, {total} bytes", flush=True)


if __name__ == "__main__":
    main()
