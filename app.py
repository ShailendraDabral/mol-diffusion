"""Gradio demo: generate 3D molecules with the trained diffusion model.

Local:   python app.py                       (uses checkpoints/best.pt, or $CKPT)
Hugging Face Space: upload this file, the src/ folder, requirements.txt and a small
checkpoint (copy checkpoints/best.pt -> best.pt, keep only the EMA weights to shrink it,
see README), then set the Space SDK to Gradio.
"""
import os
import tempfile

import gradio as gr
import numpy as np
import torch

from src.data import sample_n_atoms
from src.metrics import check_stability, to_smiles
from src.sample import generate, load_checkpoint
from src.viz import mol3d_html, to_xyz

CKPT = os.environ.get("CKPT") or next(
    (p for p in ["best.pt", "checkpoints/best.pt", "checkpoints/last.pt"] if os.path.exists(p)), None)

_STATE = {}


def _load():
    if "model" not in _STATE:
        if CKPT is None:
            raise gr.Error("No checkpoint found. Train a model first (python -m src.train ...) "
                           "or set the CKPT environment variable.")
        model, diff, hist, cfg = load_checkpoint(CKPT, device=torch.device("cpu"))
        _STATE.update(model=model, diff=diff, hist=hist)
    return _STATE["model"], _STATE["diff"], _STATE["hist"]


def generate_molecule(n_atoms, steps, sampler, seed, tries):
    """Sample `tries` molecules at once and show the best one (stable > valid > any)."""
    model, diff, hist = _load()
    n_atoms, steps, tries = int(n_atoms), int(steps), int(tries)
    if n_atoms <= 0:  # 0 = pick sizes from the QM9 size distribution
        n_atoms = int(sample_n_atoms(hist, 1, np.random.default_rng(int(seed)))[0])
    samples = generate(model, diff, [n_atoms] * tries, steps=steps,
                       ddim=(sampler == "DDIM"), seed=int(seed))

    def score(s):
        stable, n_ok, n = check_stability(*s)
        return (int(stable), int(to_smiles(*s) is not None), n_ok / n)

    best = max(samples, key=score)
    atom_idx, pos = best
    stable, n_ok, n = check_stability(atom_idx, pos)
    smiles = to_smiles(atom_idx, pos)
    n_valid = sum(to_smiles(*s) is not None for s in samples)
    status = (f"{n} atoms | {n_ok}/{n} atoms with correct valence | "
              f"{'chemically stable' if stable else 'NOT fully stable'} | "
              f"{n_valid}/{tries} samples valid")
    xyz = tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False)
    xyz.write(to_xyz(atom_idx, pos, comment=f"seed={seed} steps={steps} sampler={sampler}"))
    xyz.close()
    return mol3d_html(atom_idx, pos), smiles or "(RDKit could not parse this sample)", status, xyz.name


with gr.Blocks(title="3D Molecule Diffusion") as demo:
    gr.Markdown(
        "# 3D molecule generation with an equivariant diffusion model\n"
        "A small EDM-style model trained on QM9 (molecules with up to 9 heavy atoms of C, N, O, F). "
        "It generates atom types and 3D coordinates; bonds are inferred from distances. "
        "Fewer steps = faster but lower quality. Samples are not guaranteed to be real or useful chemistry."
    )
    with gr.Row():
        with gr.Column(scale=1):
            n_atoms = gr.Slider(0, 29, value=0, step=1, label="Number of atoms (0 = random, QM9-like)")
            steps = gr.Slider(10, 1000, value=100, step=10, label="Denoising steps")
            sampler = gr.Dropdown(["DDIM", "DDPM"], value="DDIM", label="Sampler")
            seed = gr.Number(value=0, precision=0, label="Seed")
            tries = gr.Slider(1, 16, value=8, step=1, label="Candidates to sample (best is shown)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            viewer = gr.HTML(label="3D structure")
            smiles = gr.Textbox(label="SMILES (largest fragment)", interactive=False)
            status = gr.Textbox(label="Check", interactive=False)
            xyz_file = gr.File(label="Download XYZ")
    btn.click(generate_molecule, [n_atoms, steps, sampler, seed, tries],
              [viewer, smiles, status, xyz_file])

if __name__ == "__main__":
    demo.launch()
