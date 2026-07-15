#!/usr/bin/env python
"""Split a unified (vLLM + diffusers-native) Cosmos3 checkpoint into Option A layout.

Produces a NEW root, leaving the source untouched:

    <dst>/
      model_index.json, config.json, ...   (symlinks to src)
      vae/, text_tokenizer/, ...            (symlinks to src)
      transformer/                          vLLM-clean: weight + weight_scale + input_scale
      transformer_modelopt/                 diffusers-native: + *_amax + modelopt_state.pth

vLLM auto-loads transformer/ (its hardcoded `transformer/*.safetensors` glob is
non-recursive, so the sibling transformer_modelopt/ is invisible). Diffusers loads
the native payload via `from_pretrained(root, subfolder="transformer_modelopt")`.

Requires torch + safetensors (run inside the vLLM image / an env that has them):
    python split_checkpoint_optionA.py --src <unified_root> --dst <new_root>
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Any tensor whose key contains this is diffusers-native raw quantizer state that
# vLLM's loader has no home for. The real fp8 scales live in *_scale (kept).
DROP_SUBSTR = "_quantizer."
SHARED_SKIP = {"transformer"}  # handled explicitly; everything else is symlinked


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def link_shared_root(src_root: Path, dst_root: Path) -> None:
    """Symlink every top-level entry except transformer/ (rebuilt below)."""
    for entry in sorted(src_root.iterdir()):
        if entry.name in SHARED_SKIP:
            continue
        symlink(entry, dst_root / entry.name)


def mirror_native(src_tr: Path, dst_tr: Path) -> None:
    """transformer_modelopt/ = the source transformer, verbatim (symlinks)."""
    dst_tr.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_tr.iterdir()):
        symlink(entry, dst_tr / entry.name)


def build_clean(src_tr: Path, dst_tr: Path) -> None:
    """transformer/ = source shards minus *_quantizer.* tensors; index rebuilt."""
    dst_tr.mkdir(parents=True, exist_ok=True)
    shards = sorted(src_tr.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors found in {src_tr}")

    weight_map: dict[str, str] = {}
    total_bytes = 0
    dropped = 0
    for shard in shards:
        tensors = load_file(str(shard))
        kept = {k: v for k, v in tensors.items() if DROP_SUBSTR not in k}
        dropped += len(tensors) - len(kept)
        if not kept:
            continue  # whole shard was quantizer state
        save_file(kept, str(dst_tr / shard.name), metadata={"format": "pt"})
        for k, v in kept.items():
            weight_map[k] = shard.name
            total_bytes += v.numel() * v.element_size()

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with open(dst_tr / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # config.json is fine for vLLM as-is (its quant_config expects weight_scale/
    # input_scale, which we kept). modelopt_state.pth is intentionally NOT copied.
    for extra in ("config.json",):
        src = src_tr / extra
        if src.is_file():
            symlink(src, dst_tr / extra)

    print(f"[clean] kept {len(weight_map)} tensors, dropped {dropped} quantizer tensors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unified checkpoint root")
    ap.add_argument("--dst", required=True, help="new Option-A root to create")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src).resolve(), Path(args.dst).resolve()
    src_tr = src_root / "transformer"
    if not src_tr.is_dir():
        raise SystemExit(f"missing {src_tr}")
    dst_root.mkdir(parents=True, exist_ok=True)

    link_shared_root(src_root, dst_root)
    mirror_native(src_tr, dst_root / "transformer_modelopt")
    build_clean(src_tr, dst_root / "transformer")

    print(f"[done] Option-A checkpoint at {dst_root}")
    print(f"  vLLM:     vllm serve {dst_root}")
    print(f"  diffusers: from_pretrained('{dst_root}', subfolder='transformer_modelopt')")


if __name__ == "__main__":
    main()
