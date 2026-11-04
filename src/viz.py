"""Visualization helpers: XYZ export, matplotlib grids / gifs, py3Dmol HTML for the demo."""
import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import ATOMS
from src.metrics import infer_bonds

COLORS = {"H": "#d9d9d9", "C": "#404040", "N": "#3050f8", "O": "#e02020", "F": "#50c878"}
SIZES = {"H": 25, "C": 80, "N": 80, "O": 80, "F": 70}


def to_xyz(atom_idx, pos, comment="") -> str:
    lines = [str(len(atom_idx)), comment]
    for a, p in zip(atom_idx, pos):
        lines.append(f"{ATOMS[int(a)]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    return "\n".join(lines) + "\n"


def _draw(ax, atom_idx, pos, title=None):
    pos = np.asarray(pos)
    for i, j, o in infer_bonds(atom_idx, pos):
        ax.plot(*zip(pos[i], pos[j]), color="#888888", lw=1 + o, zorder=1)
    for a, p in zip(atom_idx, pos):
        s = ATOMS[int(a)]
        ax.scatter(*p, s=SIZES[s], c=COLORS[s], edgecolors="k", linewidths=0.4, depthshade=False)
    ax.set_axis_off()
    r = max(np.abs(pos).max(), 1.0)
    ax.set_xlim(-r, r), ax.set_ylim(-r, r), ax.set_zlim(-r, r)
    if title:
        ax.set_title(title, fontsize=8)


def plot_grid(samples, path, ncols=4):
    """Save a grid of 3D molecule plots. samples: list of (atom_idx, pos)."""
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for k, (a, x) in enumerate(samples):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        _draw(ax, a, x, title=f"{len(a)} atoms")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def trajectory_gif(traj, path, n_atoms, index=0, fps=8):
    """Make a gif of one molecule denoising.

    traj: snapshots from GaussianDiffusion.sample(return_traj=True);
    n_atoms: number of real atoms of molecule `index` (drops the padding).
    """
    from PIL import Image
    import io

    frames = []
    for x, a in traj:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111, projection="3d")
        _draw(ax, a[index, :n_atoms].numpy(), x[index, :n_atoms].numpy())
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def mol3d_html(atom_idx, pos, width=480, height=380) -> str:
    """Self-contained py3Dmol viewer, wrapped in an iframe so it works inside Gradio."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(to_xyz(atom_idx, pos), "xyz")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    view.setBackgroundColor("white")
    view.zoomTo()
    inner = view._make_html()
    return (f'<iframe srcdoc="{html.escape(inner, quote=True)}" '
            f'width="{width + 20}" height="{height + 20}" style="border:0"></iframe>')
