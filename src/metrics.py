"""Chemistry-aware evaluation: bond inference, stability, validity, uniqueness, novelty.

Generated molecules are just atom types + coordinates, so bonds must be inferred from
interatomic distances. We use typical bond lengths (picometres) per atom pair, as in the
EDM codebase: a pair closer than the single-bond length (+margin) is bonded; closer than
the double/triple-bond length (+margin) gets a higher bond order.

Caveat worth stating in your write-up: this is a heuristic. Also, like EDM, RDKit adds
implicit hydrogens to under-bonded atoms during sanitization, so "validity" is lenient;
"atom/molecule stability" (exact valence match) is the stricter number.
"""
import numpy as np
from rdkit import Chem, RDLogger

from src.data import ATOMS

RDLogger.DisableLog("rdApp.*")

# typical bond lengths in picometres
BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
BONDS3 = {"C": {"C": 120, "N": 116, "O": 113}, "N": {"C": 116, "N": 110}}
MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3

ALLOWED_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}
_BOND_TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def bond_order(sym_a: str, sym_b: str, dist_angstrom: float) -> int:
    """0 = no bond, 1/2/3 = single/double/triple, from the distance alone."""
    d = 100.0 * dist_angstrom
    if d >= BONDS1[sym_a][sym_b] + MARGIN1:
        return 0
    order = 1
    if sym_a in BONDS2 and sym_b in BONDS2[sym_a] and d < BONDS2[sym_a][sym_b] + MARGIN2:
        order = 2
        if sym_a in BONDS3 and sym_b in BONDS3[sym_a] and d < BONDS3[sym_a][sym_b] + MARGIN3:
            order = 3
    return order


def infer_bonds(atom_idx, pos):
    """Return list of (i, j, order) for all bonded pairs."""
    syms = [ATOMS[int(a)] for a in atom_idx]
    n = len(syms)
    pos = np.asarray(pos)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            o = bond_order(syms[i], syms[j], float(np.linalg.norm(pos[i] - pos[j])))
            if o:
                bonds.append((i, j, o))
    return bonds


def check_stability(atom_idx, pos):
    """(molecule_stable, n_stable_atoms, n_atoms) using exact valence matching."""
    syms = [ATOMS[int(a)] for a in atom_idx]
    valence = np.zeros(len(syms), dtype=int)
    for i, j, o in infer_bonds(atom_idx, pos):
        valence[i] += o
        valence[j] += o
    stable = [valence[k] == ALLOWED_VALENCE[s] for k, s in enumerate(syms)]
    return all(stable), int(sum(stable)), len(syms)


def build_mol(atom_idx, pos):
    mol = Chem.RWMol()
    for a in atom_idx:
        mol.AddAtom(Chem.Atom(ATOMS[int(a)]))
    for i, j, o in infer_bonds(atom_idx, pos):
        mol.AddBond(i, j, _BOND_TYPE[o])
    return mol


def to_smiles(atom_idx, pos):
    """Canonical SMILES of the largest fragment, or None if RDKit rejects the molecule."""
    try:
        mol = build_mol(atom_idx, pos)
        Chem.SanitizeMol(mol)
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, key=lambda m: m.GetNumAtoms())
        largest = Chem.RemoveHs(largest)
        return Chem.MolToSmiles(largest)
    except Exception:
        return None


def is_connected(atom_idx, pos) -> bool:
    """True if the inferred bonds join all atoms into a single molecule."""
    n = len(atom_idx)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in infer_bonds(atom_idx, pos):
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)}) == 1


def train_smiles_set(mols, limit: int = 20000):
    """SMILES of (a subset of) the training molecules, for the novelty metric."""
    out = set()
    for idx, pos in mols[:limit]:
        s = to_smiles(idx, pos)
        if s:
            out.add(s)
    return out


def evaluate(samples, train_smiles=None):
    """samples: list of (atom_idx, pos). Returns a dict of metrics in [0, 1]."""
    n_mol = len(samples)
    mol_stable = atoms_stable = atoms_total = n_connected = 0
    smiles = []
    for idx, pos in samples:
        ok, ns, na = check_stability(idx, pos)
        mol_stable += int(ok)
        n_connected += int(is_connected(idx, pos))
        atoms_stable += ns
        atoms_total += na
        s = to_smiles(idx, pos)
        if s:
            smiles.append(s)
    unique = set(smiles)
    res = {
        "n_samples": n_mol,
        "atom_stability": atoms_stable / max(atoms_total, 1),
        "mol_stability": mol_stable / max(n_mol, 1),
        "validity": len(smiles) / max(n_mol, 1),
        # extra, stricter sanity number: all atoms joined into ONE molecule
        "connected": n_connected / max(n_mol, 1),
        "uniqueness": len(unique) / max(len(smiles), 1),
    }
    if train_smiles is not None:
        res["novelty"] = len(unique - train_smiles) / max(len(unique), 1)
    return res
