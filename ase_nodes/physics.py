from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import randint

from ase import Atoms
from pyiron_workflow import (
    as_function_node,
    as_macro_node,
    inputs_to_list,
    standard_nodes as std
)

from ase_nodes.atoms import ChangeChemistry, ClosestNeighbor
from ase_nodes.calculators import DelayedCalculator
from ase_nodes.filters import DelayedFilter
from ase_nodes.optimizers import DelayedOptimizer


@dataclass
class AtomicOutput:
    potential_energy: float
    forces: float


@dataclass
class MinimizeInput:
    fmax: float = 0.01
    steps: int = 10000


@as_function_node("data", "atoms")
def Static(
    atoms: Atoms,
    calculator: DelayedCalculator,
) -> tuple[AtomicOutput, Atoms]:
    atoms = atoms.copy()  # Copy for idempotency
    atoms.set_calculator(calculator.instantiate(atoms))
    data = AtomicOutput(
        potential_energy=atoms.get_potential_energy(),
        forces=atoms.get_forces()
    )
    return data, atoms.copy()


@as_macro_node("data", "atoms")
def CleanStatic(
    self,
    atoms: Atoms,
    calculator: DelayedCalculator,
    delete_after: bool = True,
):
    path = Path(".", self.semantic_path.replace(self.semantic_delimiter, "_"))
    self.go_in = std.ChangeDirectory(
        path=path
    )
    self.min = Static(
        atoms=atoms,
        calculator=calculator,
    )
    self.come_out = std.ChangeDirectory(
        path=self.go_in.outputs.started_at,
        delete_start=delete_after,
    )
    self.starting_nodes = [self.go_in]
    self.go_in >> self.min >> self.come_out
    return self.min.outputs.data, self.min.outputs.atoms


@as_function_node("data", "atoms")
def Minimize(
    atoms: Atoms,
    calculator: DelayedCalculator,
    optimizer: DelayedOptimizer,
    filter_: DelayedFilter | None = None,
    minimize_input: MinimizeInput | None = None,
) -> tuple[AtomicOutput, Atoms]:
    atoms = atoms.copy()

    atoms.set_calculator(calculator.instantiate(atoms))
    to_optimize = atoms if filter_ is None else filter_.instantiate(atoms)

    opt = optimizer.instantiate(to_optimize)
    minimize_input = MinimizeInput() if minimize_input is None else minimize_input
    opt.run(fmax=minimize_input.fmax, steps=minimize_input.steps)
    data = AtomicOutput(
        potential_energy=atoms.get_potential_energy(),
        forces=atoms.get_forces()
    )
    return data, atoms.copy()


@as_macro_node("data", "atoms")
def CleanMinimize(
    self,
    atoms: Atoms,
    calculator: DelayedCalculator,
    optimizer: DelayedOptimizer,
    filter_: DelayedFilter | None = None,
    minimize_input: MinimizeInput | None = None,
    delete_after: bool = True,
) -> tuple[AtomicOutput, Atoms]:
    path = Path(".", self.semantic_path.replace(self.semantic_delimiter, "_"))
    self.go_in = std.ChangeDirectory(
        path=path
    )
    self.min = Minimize(
        atoms=atoms,
        calculator=calculator,
        optimizer=optimizer,
        filter_=filter_,
        minimize_input=minimize_input
    )
    self.come_out = std.ChangeDirectory(
        path=self.go_in.outputs.started_at,
        delete_start=delete_after,
    )
    self.starting_nodes = [self.go_in]
    self.go_in >> self.min >> self.come_out
    return self.min.outputs.data, self.min.outputs.atoms


@as_macro_node("binding_energy", "data", "atoms")
def BindingEnergy(
    self,
    a: Atoms,
    b: Atoms,
    ab: Atoms,
    neither: Atoms,
    calculator: DelayedCalculator,
    optimizer: DelayedOptimizer,
    filter_: DelayedFilter | None = None,
    minimize_input: MinimizeInput | None = None,
) -> tuple[float, list[AtomicOutput], list[Atoms]]:
    minimizations = [
        CleanMinimize(
            label=f"min_{atoms.label}",
            parent=self,
            atoms=atoms,
            calculator=calculator,
            optimizer=optimizer,
            filter_=filter_,
            minimize_input=minimize_input,
        )
        for atoms in [a, b, ab, neither]
    ]

    self.data = inputs_to_list(4, *[m.outputs.data for m in minimizations])
    self.data.outputs.list.type_hint = list[AtomicOutput]

    self.atoms = inputs_to_list(4, *[m.outputs.atoms for m in minimizations])
    self.atoms.outputs.list.type_hint = list[Atoms]

    self.binding_energy = (
        (self.data[0].potential_energy + self.data[1].potential_energy)
        - (self.data[2].potential_energy + self.data[3].potential_energy)
    )
    return self.binding_energy, self.data, self.atoms


@as_macro_node("binding_energy", "data", "atoms")
def SoluteVacancyBinding(
    self,
    bulk: Atoms,
    solute: str,
    calculator: DelayedCalculator,
    optimizer: DelayedOptimizer,
    filter_: DelayedFilter | None = None,
    minimize_input: MinimizeInput | None = None,
    vacancy_index: int = 0,
) -> tuple[float, list[AtomicOutput], list[Atoms]]:
    self.solute_index = ClosestNeighbor(atoms=bulk, reference_index=vacancy_index)
    self.vac = ChangeChemistry(bulk, vacancy_index, None)
    self.sol = ChangeChemistry(bulk, self.solute_index, solute)
    self.solvac = ChangeChemistry(self.sol, vacancy_index, None)
    self.binding_energy = BindingEnergy(
        self.sol,
        self.vac,
        self.solvac,
        self.bulk,
        calculator=calculator,
        optimizer=optimizer,
        filter_=filter_,
        minimize_input=minimize_input,
        parent=self,  # So the semantic path exists at instantiation -- gross edge case
    )
    return (
        self.binding_energy.outputs.binding_energy,
        self.binding_energy.outputs.data,
        self.binding_energy.outputs.atoms
    )
