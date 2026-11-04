"""Data loading for 3D molecules.

Two sources:
  * "qm9"    - the real dataset (~134k molecules, up to 29 atoms incl. hydrogens).
               Needs `pip install torch_geometric`; downloads on first use and caches.
  * "smiles" - a tiny offline dataset built with RDKit (a few hundred conformers).
               Only meant for smoke tests and debugging without any download.

A molecule is a pair (atom_idx, pos): atom_idx is an int array of indices into ATOMS,
pos is an (n, 3) float32 array centered at its center of mass.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ATOMS = ["H", "C", "N", "O", "F"]
Z_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
N_TYPES = len(ATOMS)
MAX_ATOMS = 29  # QM9 maximum, including hydrogens

# Small neutral molecules made of H, C, N, O, F with <= 9 heavy atoms.
_SEED_SMILES = [
    "C", "CC", "CCC", "CCO", "CCN", "CC=O", "CC(=O)O", "CC#N", "C=C", "C#C",
    "CO", "CN", "NCC=O", "OCCO", "NCCO", "CC(C)O", "CC(C)N", "CC(C)=O", "C1CC1",
    "C1CCC1", "C1CCCC1", "C1CCCCC1", "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1",
    "Nc1ccccc1", "C1CO1", "C1CN1", "C1COC1", "OC1CC1", "CC(=O)N", "NC(=O)N",
    "CC(F)F", "FC(F)F", "CCF", "OCC#N", "C=CC=O", "C=CC#N", "CCOC", "CCNC",
    "COC(=O)C", "CC(O)C(=O)O", "NCC(=O)O", "O=CC=O", "C1=CC=CO1", "c1ccoc1",
    "c1cc[nH]c1", "c1ccncc1", "CC1CC1", "OCC1CC1", "CC(C)(C)O", "CCCCO",
]


def _center(pos: np.ndarray) -> np.ndarray:
    return (pos - pos.mean(axis=0, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- loaders
def load_qm9(root: str = "data/qm9"):
    """Load QM9 via PyTorch Geometric and cache a light-weight copy."""
    cache = os.path.join(root, "mols.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    try:
        from torch_geometric.datasets import QM9
    except ImportError as e:
        raise ImportError(
            "QM9 loading needs torch_geometric: `pip install torch_geometric`. "
            "Or run with --dataset smiles for an offline smoke test."
        ) from e
    ds = QM9(root)
    mols = []
    for d in ds:
        idx = np.array([Z_TO_IDX[int(z)] for z in d.z.tolist()], dtype=np.int64)
        mols.append((idx, _center(d.pos.numpy())))
    os.makedirs(root, exist_ok=True)
    torch.save(mols, cache)
    return mols


def load_smiles_dataset(n_conformers: int = 6, seed: int = 0):
    """Tiny offline dataset: embed seed SMILES in 3D with RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols = []
    for k, smi in enumerate(_SEED_SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conformers, randomSeed=seed + k)
        if len(cids) == 0:
            continue
        try:
            AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
        except Exception:
            pass
        idx = np.array([Z_TO_IDX[a.GetAtomicNum()] for a in m.GetAtoms()], dtype=np.int64)
        for cid in cids:
            pos = m.GetConformer(cid).GetPositions()
            mols.append((idx, _center(pos)))
    return mols


def load_molecules(name: str, root: str = "data/qm9"):
    if name == "qm9":
        return load_qm9(root)
    if name == "smiles":
        return load_smiles_dataset()
    raise ValueError(f"unknown dataset {name!r}")


def split_molecules(mols, seed: int = 0):
    """Random train/val/test split. Full QM9 -> 100k/18k/rest (as in EDM); otherwise 80/10/10."""
    rng = random.Random(seed)
    order = list(range(len(mols)))
    rng.shuffle(order)
    n = len(mols)
    if n >= 130_000:
        n_tr, n_va = 100_000, 18_000
    else:
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
    pick = lambda ids: [mols[i] for i in ids]
    return pick(order[:n_tr]), pick(order[n_tr:n_tr + n_va]), pick(order[n_tr + n_va:])


# --------------------------------------------------------------------------- torch glue
class MoleculeDataset(Dataset):
    def __init__(self, mols):
        self.mols = mols

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        return self.mols[i]


def collate(batch):
    """Pad to the largest molecule in the batch.

    Returns x (B,N,3), h (B,N,N_TYPES one-hot), mask (B,N). Padded rows are all zeros.
    """
    B = len(batch)
    N = max(len(a) for a, _ in batch)
    x = torch.zeros(B, N, 3)
    a = torch.zeros(B, N, dtype=torch.long)
    mask = torch.zeros(B, N)
    for b, (idx, pos) in enumerate(batch):
        n = len(idx)
        a[b, :n] = torch.from_numpy(idx)
        x[b, :n] = torch.from_numpy(pos)
        mask[b, :n] = 1
    h = F.one_hot(a, N_TYPES).float() * mask.unsqueeze(-1)
    return x, h, mask


def atom_count_hist(mols):
    """Distribution over number of atoms, used to pick molecule sizes at sampling time."""
    counts = {}
    for idx, _ in mols:
        counts[len(idx)] = counts.get(len(idx), 0) + 1
    total = sum(counts.values())
    return {int(n): c / total for n, c in sorted(counts.items())}


def sample_n_atoms(hist: dict, k: int, rng: np.random.Generator):
    ns = np.array(list(hist.keys()))
    ps = np.array(list(hist.values()), dtype=np.float64)
    return rng.choice(ns, size=k, p=ps / ps.sum()).tolist()
