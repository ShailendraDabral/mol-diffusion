"""Core correctness tests. Run with:  pytest -q"""
import numpy as np
import torch

from src.data import MoleculeDataset, collate, load_smiles_dataset
from src.diffusion import GaussianDiffusion
from src.egnn import EGNNDenoiser
from src.metrics import check_stability, evaluate, to_smiles
from src.utils import center_masked


def _random_rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _batch(n=4):
    mols = load_smiles_dataset(n_conformers=1)[:n]
    return collate(mols)


def test_equivariance_rotation_translation():
    """Rotating/translating the input must rotate the predicted coordinate noise the same way
    and leave the atom-type prediction unchanged."""
    torch.manual_seed(0)
    x, h, mask = _batch()
    model = EGNNDenoiser(nf=32, n_layers=3).double().eval()
    # make coordinate updates non-trivial so the test is meaningful
    for layer in model.layers:
        torch.nn.init.normal_(layer.coord_mlp[-1].weight, std=0.1)
    x, h, mask = x.double(), h.double(), mask.double()
    t = torch.full((x.shape[0],), 0.5, dtype=torch.double)
    R = _random_rotation().double()
    shift = torch.randn(1, 1, 3).double()

    ex, eh = model(x, h, t, mask)
    ex2, eh2 = model(((x @ R.T) + shift) * mask.unsqueeze(-1), h, t, mask)
    assert ex.abs().max() > 1e-6, "coordinate output is trivially zero"
    assert torch.allclose(ex2, ex @ R.T, atol=1e-6)
    assert torch.allclose(eh2, eh, atol=1e-6)


def test_permutation_equivariance():
    torch.manual_seed(0)
    x, h, mask = _batch(2)
    model = EGNNDenoiser(nf=32, n_layers=2).double().eval()
    x, h, mask = x.double(), h.double(), mask.double()
    t = torch.full((2,), 0.3, dtype=torch.double)
    n = int(mask[0].sum())
    perm = torch.cat([torch.randperm(n), torch.arange(n, x.shape[1])])
    ex, eh = model(x, h, t, mask)
    ex_p, eh_p = model(x[:, perm], h[:, perm], t, mask[:, perm])
    assert torch.allclose(ex[:, perm], ex_p, atol=1e-6)
    assert torch.allclose(eh[:, perm], eh_p, atol=1e-6)


def test_padding_does_not_leak():
    """A molecule's output must not depend on how much padding surrounds it."""
    torch.manual_seed(0)
    mols = load_smiles_dataset(n_conformers=1)
    small = min(mols, key=lambda m: len(m[0]))
    big = max(mols, key=lambda m: len(m[0]))
    model = EGNNDenoiser(nf=32, n_layers=2).eval()
    for layer in model.layers:
        torch.nn.init.normal_(layer.coord_mlp[-1].weight, std=0.1)
    t = torch.tensor([0.5])
    x1, h1, m1 = collate([small])
    x2, h2, m2 = collate([small, big])
    ex1, eh1 = model(x1, h1, t, m1)
    ex2, eh2 = model(x2, h2, torch.tensor([0.5, 0.5]), m2)
    n = len(small[0])
    assert torch.allclose(ex1[0, :n], ex2[0, :n], atol=1e-5)
    assert torch.allclose(eh1[0, :n], eh2[0, :n], atol=1e-5)
    assert ex2[0, n:].abs().max() == 0


def test_noise_is_zero_com_and_loss_finite():
    x, h, mask = _batch()
    diff = GaussianDiffusion(T=50)
    t = torch.randint(0, 50, (x.shape[0],))
    _, _, ex, eh = diff.q_sample(x, h, t, mask)
    com = (ex * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    assert com.abs().max() < 1e-5
    assert (center_masked(x, mask) - x).abs().max() < 1e-4  # dataset already centered
    loss = diff.loss(EGNNDenoiser(nf=16, n_layers=2), x, h, mask)
    assert torch.isfinite(loss)


def test_schedule_is_monotone_and_floored():
    """abar must decrease and stay >= the floor, otherwise sampling blows up at the first step."""
    for sched in ("cosine", "linear"):
        d = GaussianDiffusion(T=1000, schedule=sched)
        assert (d.abar[1:] <= d.abar[:-1] + 1e-12).all()
        assert d.abar[-1] >= 1e-4 - 1e-9 and d.abar[0] < 1.0


def test_sampler_recovers_data_with_oracle_denoiser():
    """If the network were perfect, the sampler must reproduce the data exactly.
    This isolates sampler maths (DDIM and DDPM, strided) from model quality."""
    x0, h0, mask = _batch(3)
    diff = GaussianDiffusion(T=200)
    h0s = h0 * diff.h_scale

    class Oracle(torch.nn.Module):
        def forward(self, x, h, t, mask):
            ab = diff.abar[(t * diff.T - 1).round().long()].view(-1, 1, 1)
            return (x - ab.sqrt() * x0) / (1 - ab).sqrt(), (h - ab.sqrt() * h0s) / (1 - ab).sqrt()

    n = [int(m) for m in mask.sum(1)]
    for ddim in (True, False):
        x, a, _ = diff.sample(Oracle(), n, steps=40, ddim=ddim)
        assert (x - x0).abs().max() < 1e-3
        assert ((a == h0.argmax(-1)) * mask).sum() == mask.sum()


def test_sampler_shapes_and_centering():
    diff = GaussianDiffusion(T=20)
    model = EGNNDenoiser(nf=16, n_layers=2).eval()
    for ddim in (False, True):
        x, a, mask = diff.sample(model, [5, 8, 3], steps=7, ddim=ddim)
        assert x.shape == (3, 8, 3) and a.shape == (3, 8)
        assert torch.isfinite(x).all()
        assert (x * (1 - mask).unsqueeze(-1)).abs().max() == 0
        com = x.sum(1) / mask.sum(1, keepdim=True)
        assert com.abs().max() < 1e-4


def test_real_molecules_are_stable_and_valid():
    """Sanity check of the bond-inference heuristic on RDKit conformers of real molecules."""
    mols = load_smiles_dataset(n_conformers=1)
    res = evaluate(mols)
    assert res["atom_stability"] > 0.9, res
    assert res["mol_stability"] > 0.7, res
    assert res["validity"] > 0.9, res


def test_known_molecule_smiles():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    from src.data import Z_TO_IDX
    idx = np.array([Z_TO_IDX[a.GetAtomicNum()] for a in m.GetAtoms()])
    pos = m.GetConformer().GetPositions()
    assert check_stability(idx, pos)[0]
    assert to_smiles(idx, pos) == "CCO"
    # a lone floating atom set is not stable
    assert not check_stability(idx, pos * 3.0)[0]
