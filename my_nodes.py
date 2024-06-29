from __future__ import annotations

import sys

from ase import Atoms
from pyiron_workflow import (
    as_function_node,
    as_macro_node,
    for_node,
    standard_nodes as std
)
import ase_nodes as an


@as_macro_node("biggest_species")
def BiggestSpeciesGuess(self, species: list[str]) -> str:
    self.spec_vol_df = for_node(
        an.atoms.PerAtomVolume,
        iter_on=("species",),
        species=species
    )
    self.max_index = std.PureCall(
        self.spec_vol_df.outputs.df["per_atom_volume"].idxmax
    )
    return species[self.max_index]


@as_function_node("below_threshold")
def AbsoluteDifferenceBelowThreshold(a, b, threshold) -> bool:
    return bool(abs(a - b) < threshold)


@as_macro_node("repeats")
def IncreaseSizeWhileEnergyChanges(
    self,
    host_unit_cell: Atoms,
    solute: str,
    calculator: an.calculators.DelayedCalculator,
    optimizer: an.optimizers.DelayedOptimizer,
    filter_: an.filters.DelayedFilter | None = None,
    energy_threshold: float = 3e-4,
) -> int:
    self.repeats = std.Add(0, 1)
    self.repeats.inputs.obj = self.repeats  # Loop on itself

    self.supercell = an.atoms.Repeat(host_unit_cell, self.repeats)

    self.binding = an.physics.SoluteVacancyBinding(
        bulk=self.supercell,
        solute=solute,
        calculator=calculator,
        optimizer=optimizer,
        filter_=filter_,
        parent=self,  # So the semantic path exists at instantiation -- gross edge case
    )

    self.binding_history = std.AppendToList()
    self.binding_history.inputs.existing = self.binding_history  # Loop on itself
    self.binding_history.inputs.new_element = sys.float_info.max  # Initial value
    self.binding_history.inputs.new_element = self.binding.outputs.binding_energy

    self.last_binding = self.binding_history[-1]

    self.threshold = AbsoluteDifferenceBelowThreshold(
        self.binding.outputs.binding_energy,
        self.last_binding,
        energy_threshold,
    )

    self.switch = std.If(condition=self.threshold)

    self.starting_nodes = [self.repeats]
    (
        self.repeats >>
        self.supercell >>
        self.binding_history >>
        self.binding >>
        self.last_binding >>
        self.threshold >>
        self.switch
    )
    self.switch.signals.output.false >> self.repeats
    return self.repeats


@as_macro_node("binding_energy", "data", "atoms")
def LAMMPSSoluteVacancyBinding(
    self,
    bulk: Atoms,
    solute: str,
    optimizer: an.optimizers.DelayedOptimizer,
    filter_: an.filters.DelayedFilter | None = None,
    minimize_input: an.physics.MinimizeInput | None = None,
    vacancy_index: int = 0,
) -> tuple[float, list[an.physics.AtomicOutput], list[Atoms]]:
    """
    Unlike EMT and Espresso, the LAMMPS calculator has settings (the potential) which depend
    strongly on the chemistry of the atoms being calculated.

    To make sure we use the same potential across all four components of the binding energy
    calculation, we make sure this is pre-set before calling the binding energy node.
    """
    self.host_species_list = std.PureCall(
        bulk.get_chemical_symbols
    )
    self.all_species = std.AppendToList(
        existing=self.host_species_list,
        new_element=solute
    )
    self.lammps_potential = an.calculators.IprpyLAMPPSPotentials(self.all_species)
    self.potential_params = an.calculators.LAMMPSPotentialsToParams(
        self.lammps_potential,
        choice=0
    )
    self.calc = an.calculators.LAMMPS(potential_parameters=self.potential_params)
    self.binding = an.physics.SoluteVacancyBinding(
        bulk=bulk,
        solute=solute,
        calculator=self.calc,
        optimizer=optimizer,
        filter_=filter_,
        minimize_input=minimize_input,
        vacancy_index=vacancy_index,
        parent=self,  # So the semantic path exists at instantiation -- gross edge case
    )
    return (
        self.binding.outputs.binding_energy,
        self.binding.outputs.data,
        self.binding.outputs.atoms
    )
