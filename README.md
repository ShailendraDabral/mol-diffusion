# 3D molecule generation with an equivariant diffusion model

A small, readable re-implementation of **EDM** (Hoogeboom et al., 2022,
*Equivariant Diffusion for Molecule Generation in 3D*) on **QM9**, sized for a free
Kaggle/Colab GPU, with a Gradio demo you can host on Hugging Face Spaces.

The model generates a molecule as *atom types + 3D coordinates*. Bonds are inferred from
distances afterwards and checked with RDKit.

> Fill this table in with your own results after training (see "Results" below).

| Metric | Your model | Notes |
| --- | --- | --- |
| Atom stability | _TBD_ | % atoms with the correct valence |
| Molecule stability | _TBD_ | % molecules where every atom is stable |
| Validity | _TBD_ | RDKit sanitizes the molecule (lenient, see caveats) |
| Connected | _TBD_ | all atoms form one molecule |
| Uniqueness | _TBD_ | distinct SMILES among valid |
| Novelty | _TBD_ | valid + unique + not in training set |

## Layout

```
src/data.py       QM9 loading (via torch_geometric), tiny offline RDKit dataset, padding/masks
src/egnn.py       E(3)-equivariant denoiser (dense EGNN)
src/diffusion.py  forward process, loss, DDPM + DDIM samplers (zero-COM noise)
src/train.py      training loop with EMA, resume, periodic evaluation
src/sample.py     generate + evaluate + save xyz / metrics / image grid
src/metrics.py    bond inference, atom/molecule stability, validity, uniqueness, novelty
src/viz.py        matplotlib grids / gifs, py3Dmol HTML for the demo
src/export.py     shrink a checkpoint (EMA only) for the demo
app.py            Gradio demo
tests/            equivariance, padding, noise, sampler and metric tests
configs/          small.yaml (free GPU) and full.yaml (paper-ish scale)
notebooks/        run_on_kaggle_colab.ipynb
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                  # ~5 s, no data needed

# 1) offline smoke test (no download, CPU is fine)
python -m src.train --dataset smiles --epochs 3 --nf 32 --layers 2 --workers 0 \
    --eval-every 1 --eval-samples 20 --eval-steps 10 --ckpt-dir /tmp/smoke

# 2) the sanity test from the plan: overfit 32 molecules
python -m src.train --dataset smiles --subset 32 --epochs 1500 --nf 64 --layers 4 \
    --lr 5e-4 --ema-decay 0.99 --batch-size 32 --workers 0 --eval-every 500

# 3) real training on QM9 (GPU). Downloads QM9 on first run.
python -m src.train --config configs/small.yaml

# 4) evaluate and sample (use --ddim --steps 100..250 for speed)
python -m src.sample --ckpt checkpoints/best.pt --n 1000 --steps 250 --ddim --novelty

# 5) demo
python -m src.export --ckpt checkpoints/best.pt --out best.pt
python app.py
```

Training resumes automatically from `checkpoints/last.pt`, so you can chain several
free GPU sessions.

## How it works (one paragraph)

Each molecule is N atoms with positions `x` (N x 3) and one-hot types `h` (N x 5). We add
Gaussian noise to both and train an EGNN to predict that noise (the same simplified DDPM
loss as for images). Two details are specific to molecules: coordinate noise is projected to
have **zero center of mass** (so translation is not a free variable), and the denoiser is
**E(3)-equivariant** (it only uses pairwise distances and moves atoms along relative
vectors), so rotating the input rotates the output. Sampling starts from noise and denoises
step by step; atom types come from the argmax of the final features.

## Things I hit while building this (worth knowing)

* **Floor on alpha-bar.** The raw cosine schedule ends around 1e-9. The first sampling step
  divides the network's noise error by sqrt(alpha-bar) (about 2e4), and coordinates explode to
  thousands of angstroms. Image models hide this by clipping x0 to [-1, 1]; molecules have no
  such clip. `GaussianDiffusion(min_abar=1e-4)` fixes it. A test guards it.
* **Aggregation scale.** With `norm_factor=100` the coordinate update is too small to ever
  reproduce the identity-like noise at high noise levels, and the loss plateaus. EDM's QM9
  setting is `norm_factor=1`, which is the default here.
* **EMA lag on tiny runs.** With decay 0.999 and only ~1000 steps, EMA weights are still
  mostly the random init. Use ~0.99 for tiny debugging runs, 0.999 for real ones.

## Caveats to state honestly in your write-up

* Bonds are inferred from a bond-length table, a heuristic. Stability is the strict metric.
* "Validity" follows EDM: RDKit adds implicit hydrogens to under-bonded atoms and we keep only
  the largest fragment, so it is lenient. That is why a `connected` metric is reported too.
* QM9 molecules are tiny (at most 9 heavy atoms) and not drug-like. A stable molecule is not
  necessarily synthesizable or useful.
* Results on a free GPU will be below the paper's. Report your own numbers next to the paper's.
* The code was tested with unit tests, an oracle-denoiser sampler test, and short CPU runs on an
  offline RDKit dataset. It has **not** been trained on real QM9 yet, so check your first real run
  carefully (loss should fall; stability metrics should climb over epochs).

## Ideas to extend

Conditional generation on a QM9 property (feed a normalized HOMO-LUMO gap as an extra input),
flow matching instead of DDPM, a latent version (autoencoder + diffusion), scaffold inpainting,
GEOM-Drugs for larger molecules.

## Deploying the demo on Hugging Face Spaces

1. Create a Space with the **Gradio** SDK.
2. Upload `app.py`, `src/`, `requirements.txt` (you can drop `torch_geometric`) and the
   exported `best.pt` (about 0.5 MB for the small config).
3. The 3D viewer loads 3Dmol.js from a CDN, so the Space needs internet access (default).

## References

Ho et al. 2020 (DDPM) · Satorras et al. 2021 (EGNN) · Hoogeboom et al. 2022 (EDM) ·
Xu et al. 2023 (GeoLDM) · Vignac et al. 2023 (DiGress)
