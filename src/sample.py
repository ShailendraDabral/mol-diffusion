"""Load a checkpoint, generate molecules, evaluate and save them.

    python -m src.sample --ckpt checkpoints/best.pt --n 1000 --steps 250 --ddim
"""
import argparse
import json
import os

import numpy as np
import torch

from src.data import sample_n_atoms
from src.diffusion import GaussianDiffusion
from src.egnn import EGNNDenoiser
from src.metrics import evaluate, train_smiles_set
from src.utils import get_device
from src.viz import plot_grid, to_xyz


def load_checkpoint(path, device=None):
    """Returns (model, diffusion, atom_hist, config). Uses the EMA weights."""
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = EGNNDenoiser(nf=cfg["nf"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = GaussianDiffusion(cfg["T"], cfg["schedule"], cfg["h_scale"]).to(device)
    return model, diff, ck["atom_hist"], cfg


@torch.no_grad()
def generate(model, diff, n_atoms, steps=None, ddim=False, seed=None, return_traj=False):
    """Run the sampler and return a list of (atom_idx ndarray, pos ndarray) per molecule."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=diff.betas.device).manual_seed(int(seed))
    out = diff.sample(model, n_atoms, steps=steps, ddim=ddim, return_traj=return_traj,
                      generator=gen)
    (x, a, mask), traj = (out if return_traj else (out, None))
    samples = []
    for b, n in enumerate(n_atoms):
        samples.append((a[b, :n].cpu().numpy(), x[b, :n].cpu().numpy()))
    return (samples, traj) if return_traj else samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--steps", type=int, default=None, help="denoising steps (default: all T)")
    p.add_argument("--ddim", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--novelty", action="store_true", help="also compute novelty (needs dataset)")
    p.add_argument("--dataset", default=None, help="dataset for novelty: qm9 | smiles")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    model, diff, hist, cfg = load_checkpoint(args.ckpt)
    rng = np.random.default_rng(args.seed)
    samples = []
    for start in range(0, args.n, args.batch_size):
        k = min(args.batch_size, args.n - start)
        ns = sample_n_atoms(hist, k, rng)
        samples += generate(model, diff, ns, args.steps, args.ddim, seed=args.seed + start)

    train_smiles = None
    if args.novelty:
        from src.data import load_molecules, split_molecules
        mols = load_molecules(args.dataset or cfg.get("dataset", "qm9"), cfg.get("data_root", "data/qm9"))
        train_smiles = train_smiles_set(split_molecules(mols)[0])
    metrics = evaluate(samples, train_smiles)
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "samples.xyz"), "w") as f:
        for i, (a, x) in enumerate(samples):
            f.write(to_xyz(a, x, comment=f"sample {i}"))
    plot_grid(samples[:16], os.path.join(args.out, "grid.png"))
    print(f"saved metrics.json, samples.xyz, grid.png to {args.out}/")


if __name__ == "__main__":
    main()
