"""Shrink a training checkpoint to just what the demo needs (EMA weights + config).

    python -m src.export --ckpt checkpoints/best.pt --out best.pt
Drops the optimizer state and raw weights, so the file is ~3x smaller.
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="best.pt")
    args = p.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    light = {"ema": ck["ema"], "config": ck["config"], "atom_hist": ck["atom_hist"]}
    torch.save(light, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
