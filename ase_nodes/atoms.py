from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from pyiron_workflow import as_function_node, as_macro_node, standard_nodes as std


# ASE has all sorts of parameters like `a: float = None`, which causes the node's type
# checking to break
@as_function_node("atoms")
def Bulk(
    name: str,
    crystalstructure: str | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    *,
    alpha: float | None = None,
    covera: float | None = None,
    u: float | None = None,
    orthorhombic: bool = False,
    cubic: bool = False,
    basis=None,
) -> Atoms:
    return bulk(
        name=name,
        crystalstructure=crystalstructure,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        covera=covera,
        u=u,
        orthorhombic=orthorhombic,
        cubic=cubic,
        basis=basis,
    )
Bulk.__doc__ = bulk.__doc__


@as_function_node("atoms")
def ChangeChemistry(
    atoms: Atoms,
    indices: int | list[int],
    new_species: str | None,
    idempotent: bool = True
) -> Atoms:
    """Change the chemistry of one or more atom (or create vacancy(ies) with `None`)."""
    if idempotent:
        atoms = atoms.copy()

    indices = [indices] if isinstance(indices, int) else indices

    if new_species is None:
        to_del = sorted(indices)[::-1]  # Reverse to avoid messing up our index
        for i in to_del:
            atoms.pop(i)
    else:
        symbols = atoms.get_chemical_symbols()
        for i in indices:
            symbols[i] = new_species
        atoms.set_chemical_symbols(symbols)
    return atoms


@as_function_node("atoms")
def Repeat(atoms: Atoms, repeats: int, idempotent: bool = True) -> Atoms:
    """It's a bird, it's a plane, it's SuperCell!"""
    if idempotent:
        atoms = atoms.copy()
    return atoms.repeat(rep=repeats)


@as_macro_node("per_atom_volume")
def PerAtomVolume(self, species: str) -> float:
    """ASE's guess at the per-atom volume of the chemical species"""
    self.unit = Bulk(species)
    self.cell_volume = std.PureCall(self.unit.get_volume)
    self.n_atoms = std.Length(self.unit)
    return self.cell_volume / self.n_atoms


@as_function_node("index")
def ClosestNeighbor(atoms: Atoms, reference_index: int = 0) -> int:
    """The index for (one of) the site(s) closest to the reference site."""
    displacement = atoms.positions - atoms.positions[reference_index]
    # We intentionally avoid any explicit mention of numpy (even though these
    # positions are a numpy.ndarray) so we don't need to explicitly add it to our
    # dependencies (although obviously it comes indirectly)
    distance = (displacement * displacement).sum(axis=1) ** 0.5
    return int(distance.argsort()[1])
