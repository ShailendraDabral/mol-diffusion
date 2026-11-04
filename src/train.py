"""Train the equivariant diffusion model.

    python -m src.train --config configs/small.yaml            # real QM9 on a free GPU
    python -m src.train --dataset smiles --epochs 2 --nf 32    # offline smoke test (CPU ok)

Checkpoints (model, EMA, optimizer, epoch) are written every epoch and the run resumes
automatically from checkpoints/last.pt, so a dropped Colab/Kaggle session loses little.
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (MoleculeDataset, atom_count_hist, collate, load_molecules,
                      sample_n_atoms, split_molecules)
from src.diffusion import GaussianDiffusion
from src.egnn import EGNNDenoiser
from src.metrics import evaluate
from src.sample import generate
from src.utils import EMA, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="yaml file with defaults")
    p.add_argument("--dataset", default="qm9", choices=["qm9", "smiles"])
    p.add_argument("--data-root", dest="data_root", default="data/qm9")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nf", type=int, default=128, help="hidden size")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--h-scale", dest="h_scale", type=float, default=0.25)
    p.add_argument("--ema-decay", dest="ema_decay", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0, help="gradient clipping norm")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=5)
    p.add_argument("--eval-samples", dest="eval_samples", type=int, default=200)
    p.add_argument("--eval-steps", dest="eval_steps", type=int, default=100)
    p.add_argument("--ckpt-dir", dest="ckpt_dir", default="checkpoints")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None,
                   help="stop after this many optimizer steps (debugging)")
    p.add_argument("--subset", type=int, default=None,
                   help="train on only the first N molecules (use 32 for the overfit sanity test)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    # two-pass parse so YAML values become defaults that CLI flags can still override
    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            p.set_defaults(**yaml.safe_load(f))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    mols = load_molecules(args.dataset, args.data_root)
    train, val, _ = split_molecules(mols, args.seed)
    if args.subset:
        train = train[:args.subset]
    hist = atom_count_hist(train)
    print(f"molecules: {len(train)} train / {len(val)} val")
    loader = DataLoader(MoleculeDataset(train), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.workers, drop_last=len(train) > args.batch_size)

    model = EGNNDenoiser(nf=args.nf, n_layers=args.layers).to(device)
    diff = GaussianDiffusion(args.T, args.schedule, args.h_scale).to(device)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    config = {k: getattr(args, k) for k in
              ["nf", "layers", "T", "schedule", "h_scale", "dataset", "data_root"]}
    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    start_epoch, best_score, step = 0, -1.0, 0

    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_score, step = ck["epoch"] + 1, ck["best_score"], ck["step"]
        print(f"resumed from {last_path} at epoch {start_epoch}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="mol-diffusion", config=vars(args))

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "step": step, "best_score": best_score, "config": config,
                    "atom_hist": hist}, path)

    done = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, h, mask in loader:
            x, h, mask = x.to(device), h.to(device), mask.to(device)
            loss = diff.loss(model, x, h, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ema.update(model)
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        msg = f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | {time.time() - t0:.0f}s"
        log = {"epoch": epoch, "loss": float(np.mean(losses))}

        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs or done):
            rng = np.random.default_rng(args.seed)
            ns = sample_n_atoms(hist, args.eval_samples, rng)
            samples = generate(ema.model, diff, ns, steps=args.eval_steps, ddim=True, seed=args.seed)
            m = evaluate(samples)
            msg += (f" | atom_stab {m['atom_stability']:.3f} mol_stab {m['mol_stability']:.3f} "
                    f"valid {m['validity']:.3f} uniq {m['uniqueness']:.3f}")
            log.update(m)
            if m["mol_stability"] + m["atom_stability"] > best_score:
                best_score = m["mol_stability"] + m["atom_stability"]
                save(best_path, epoch)
        print(msg, flush=True)
        if run:
            run.log(log)
        save(last_path, epoch)
        if done:
            break

    if not os.path.exists(best_path):  # no eval ran: make sure a usable checkpoint exists
        save(best_path, epoch)
    print("done.")


if __name__ == "__main__":
    main()
