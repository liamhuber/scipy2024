from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from functools import partialmethod
from pathlib import Path
import os
import sys

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.calculators.emt import EMT as ASEEMT
from ase.calculators.espresso import Espresso as ASEEspresso, EspressoProfile
from ase.calculators.lammpsrun import LAMMPS as ASELAMMPS
from pandas import DataFrame, read_csv
from pyiron_workflow import as_function_node

from ase_nodes.shared import DelayedInstantiator


class DelayedCalculator(DelayedInstantiator, ABC):
    def instantiate(self, atoms: Atoms) -> BaseCalculator:
        args, kwargs = self._parse_atoms(atoms)
        return super().instantiate(*args, **kwargs)

    @abstractmethod
    def _parse_atoms(self, atoms: Atoms) -> tuple[tuple, dict]:
        """
        Ensure that the atoms are consistent with the calculator representation.

        Return new args and additional kwargs to supply at instantiation as needed.
        """


DelayedCalculator.__init__.__annotations__.update({"cls": type[BaseCalculator]})


def _species_to_set(
    species: str | list[str] | tuple[str, ...] | set[str] | Atoms
) -> set[str]:
    if isinstance(species, str):
        species = {species}
    elif isinstance(species, (list, tuple)):
        species = set(species)
    elif isinstance(species, set):
        pass
    elif isinstance(species, Atoms):
        species = set(species.get_chemical_symbols())
    else:
        raise TypeError(
            f"Expected to be able to convert `species` into a set of strings, but got "
            f"{species}"
        )
    return species


def _maybe_mpi(
    mpi_command: str | None,
    n_cores: int,
    executable: str,
    oversubscribe: bool = False,
    use_hwthread_cpus: bool = False,
) -> str:
    return (
        executable if mpi_command is None
        else f"{mpi_command} "
             f"-n {n_cores} "
             f"{'--oversubscribe ' if oversubscribe else ''} "
             f"{'--use-hwthread-cpus ' if use_hwthread_cpus else ''}"
             f"{executable} "
    )


class DelayedEMT(DelayedCalculator):

    allowed_species = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"}

    __init__ = partialmethod(DelayedCalculator.__init__, ASEEMT)

    def _parse_atoms(self, atoms: Atoms) -> tuple[tuple, dict]:
        species = _species_to_set(atoms)
        disallowed = species.difference(self.allowed_species)
        if len(disallowed) > 0:
            raise ValueError(
                f"{ASEEMT.__name__} only supports {self.allowed_species}, but "
                f"{disallowed} was supplied."
            )
        return (), {}


@as_function_node("calculator")
def EMT(
    asap_cutoff: bool = False
) -> DelayedCalculator:
    return DelayedEMT(cls_kwargs={"asap_cutoff": asap_cutoff})
EMT.__doc__ = ASEEMT.__doc__
EMT.allowed_species = DelayedEMT.allowed_species  # Convenience access to this info


def _get_pseudo_dojo_dir() -> str:
    """
    Get the path to the pseudo-dojo pseudopotentials directory accompanying this code.
    """
    repo_dir = Path(__file__).parent.parent
    pseudo_dir = repo_dir.joinpath("resources", "nc-sr-05_pbe_standard_upf")
    return str(pseudo_dir.resolve())


def _get_pseudo_dojo_dict(pseudopotentials_directory) -> dict:
    """
    Assumes that all the potentials in the directory are `"X.upf"`, as from
    pseudo-dojo.org.
    """
    pseudopotentials = {}
    for filename in os.listdir(pseudopotentials_directory):
        if (
            os.path.isfile(os.path.join(pseudopotentials_directory, filename))
            and filename.endswith(".upf")
        ):
            key = filename.split('.')[0]
            pseudopotentials[key] = filename
    return pseudopotentials


class DelayedEspresso(DelayedCalculator):

    __init__ = partialmethod(DelayedCalculator.__init__, ASEEspresso)

    def _parse_atoms(self, atoms: Atoms) -> tuple[tuple, dict]:
        species = _species_to_set(atoms)
        missing = species.difference(self.cls_kwargs["pseudopotentials"].keys())
        if len(missing) > 0:
            raise ValueError(
                f"{ASEEspresso.__name__} requires pseudopotentials to be specified "
                f"for each species, but no potential(s) for {missing} was supplied."
            )
        return (), {}


@as_function_node("calculator", validate_output_labels=False)
def Espresso(
    pseudopotentials_directory: str | None = None,
    pseudopotentials: dict[str, str] | None = None,
    kpoints: tuple[int, int, int] = (1, 1, 1),
    input_data: dict | None = None,
    mpi_command: str | None = "mpiexec",
    n_cores: int = 1,
    oversubscribe: bool = False,
    use_hwthread_cpus: bool = False,
    executable: str = "pw.x",
    working_directory: str = "."
) -> DelayedCalculator:
    pseudopotentials_directory = (
        _get_pseudo_dojo_dir() if pseudopotentials_directory is None
        else pseudopotentials_directory
    )
    profile = EspressoProfile(
        command=_maybe_mpi(
            mpi_command, n_cores, executable, oversubscribe, use_hwthread_cpus
        ),
        pseudo_dir=pseudopotentials_directory
    )

    if pseudopotentials is None:
        pseudopotentials = _get_pseudo_dojo_dict(pseudopotentials_directory)

    input_data = {} if input_data is None else input_data
    default_input_data = {
        "tprnfor": True,
        "ecutwfc": 60,
        "occupations": "smearing",
        "degauss": 0.01,
    }
    default_input_data.update(input_data)

    return DelayedEspresso(
        cls_kwargs={
            "profile": profile,
            "pseudopotentials": pseudopotentials,
            "kpoints": kpoints,
            "input_data": default_input_data,
            "directory": working_directory,
        }
    )


def _iprpy_conda_data_path() -> Path:
    conda_env_path = Path(sys.executable).parent.parent
    return conda_env_path.joinpath("share", "iprpy")


def _get_iprpy_lammps_potentials_df() -> DataFrame:
    df = read_csv(
        _iprpy_conda_data_path().joinpath("potentials_lammps.csv"),
        index_col=0,
        converters={
            "Species": lambda x: x.replace("'", "")
            .strip("[]")
            .split(", "),
            "Config": lambda x: x.replace("'", "")
            .replace("\\n", "\n")
            .strip("[]")
            .split(", "),
            "Filename": lambda x: x.replace("'", "")
            .strip("[]")
            .split(", "),
        },
    )  # Converters copied and pasted from pyiron_atomistics
    return df


def _filter_iprpy_lammps_potentials(
    df: DataFrame,
    species: str | list[str] | tuple[str, ...] | set[str] | Atoms | None = None,
    excluded_styles: tuple[str, ...] = ("bop", "kim")
):
    species = _species_to_set(species)
    has_all_species = [
        len(species.difference(represented)) == 0
        for represented in df["Species"].values
    ]
    accepted_style = [
        not any(
            config[0].startswith(f"pair_style {style}")
            for style in excluded_styles
        )
        for config in df["Config"].values
    ] if len(excluded_styles) > 0 else len(has_all_species) * True
    good_row = [
        spec and style for (spec, style) in zip(has_all_species, accepted_style)
    ]
    relevant_df = df[good_row]
    if len(relevant_df) == 0:
        raise ValueError(f"Could not find a potential for {species}")
    return relevant_df


@as_function_node("potentials")
def IprpyLAMPPSPotentials(
    species: str | list[str] | tuple[str, ...] | set[str] | Atoms | None = None,
    excluded_styles: tuple[str, ...] | list[str] = ("bop", "kim"),
) -> DataFrame:
    """
    Load the `iprpy` LAMMPS potential dataframe and filter it by species and style.

    Args:
        species (str | list[str] | tuple[str, ...] | set[str] | Atoms | None): Chemical
            symbols to filter for. Defaults to None, no filter.
        excluded_styles (tuple[str]): Styles to filter for. Excludes "bop" and "kim" by
            default.

    Returns:
        DataFrame: Information about the available potentials.
    """
    return _filter_iprpy_lammps_potentials(
        _get_iprpy_lammps_potentials_df(),
        species=species,
        excluded_styles=excluded_styles
    )


@dataclass
class LammpsPotentialParameters:
    pair_style: str
    pair_coeff: list[str]
    files: list[str]
    specorder: str

    @classmethod
    @property
    def key_set(cls) -> set[str]:
        return set(f.name for f in fields(cls))

    @classmethod
    def key_overlap(cls, d: dict) -> set[str]:
        return cls.key_set.intersection(d.keys())

    def asdict(self) -> dict:
        return asdict(self)


@as_function_node()
def LAMMPSPotentialsToParams(
    df: DataFrame,
    choice: int = 0,
) -> LammpsPotentialParameters:
    """
    Choose a row of a dataframe formatted like the `iprpy` lammps potentials dataframe
    and convert it into a dict of parameters suitable for an ase lammps calculator.

    Args:
        df (DataFrame): The dataframe to read from.
        choice (int): The row (not index!) to choose.

    Returns:

    """
    id_ = df.index[choice]
    config = df["Config"][id_]
    filenames = df["Filename"][id_]
    return LammpsPotentialParameters(
        pair_style=config[0].replace("\n", "").replace("pair_style ", ""),
        pair_coeff=[
            pair.replace("pair_coeff ", "")
            for c in config[1:]
            for pair in c.split("\n")[:-1]
        ],
        files=[str(_iprpy_conda_data_path().joinpath(f)) for f in filenames if f != ""],
        specorder=df["Species"][id_]
    )


def iprpy_lammps_potential_ase_params(
    species: str | list[str] | tuple[str, ...] | set[str] | Atoms | None,
    excluded_styles: tuple[str, ...] | list[str] = ("bop", "kim"),
    choice: int = 0,
) -> LammpsPotentialParameters:
    return LAMMPSPotentialsToParams.node_function(
        IprpyLAMPPSPotentials.node_function(species, excluded_styles),
        choice=choice,
    )


class DelayedLammps(DelayedCalculator):

    __init__ = partialmethod(DelayedCalculator.__init__, ASELAMMPS)

    def _parse_atoms(self, atoms: Atoms) -> tuple[tuple, dict]:
        keys_provided = LammpsPotentialParameters.key_overlap(self.cls_kwargs)
        if len(keys_provided) == 0:
            # Automatically build from atoms using the first available potential
            new_kwargs = iprpy_lammps_potential_ase_params(
                _species_to_set(atoms)
            ).asdict()
        elif len(keys_provided) == 4:
            # Take whatever was provided by the user without verification or update
            new_kwargs = {}
        else:
            raise ValueError(
                f"{self.__class__.__name__} expected all of "
                f"{LammpsPotentialParameters.key_set} to be available or none of them "
                f"(in which case they are generated automatically based on the atoms "
                f"provided), but found {keys_provided}"
            )
        return (), new_kwargs


@as_function_node("calculator", validate_output_labels=False)
def LAMMPS(
    ase_kwargs: dict | None = None,
    potential_parameters: LammpsPotentialParameters | None = None,
    mpi_command: str | None = "mpiexec",
    n_cores: int = 1,
    oversubscribe: bool = False,
    use_hwthread_cpus: bool = False,
    executable: str = "lmp",
    working_directory: str = "."
) -> DelayedCalculator:
    """
    Create an ASE LAMMPS calculator configured for a particular model representation.

    Args:
        ase_kwargs (dict | None): C.f. :class:`ase.calculators.lammpsrun.LAMMPS`.
            Defaults to `None` -- just use the ASE defaults.
        potential_parameters (LammpsPotentialParameters | None): An optional data
            object for providing a specification of the potential with which to
            represent the atoms; when provided it must not overlap with
            :param:`ase_kwargs`.
        mpi_command (str): The MPI executable to use. If `None`, the executable is
            used directly.
        n_cores (int): How many cores to run on.
        oversubscribe (bool): Whether to add `--oversubscribe` to the mpi call.
        use_hwthread_cpus (bool): Whether to add `--use-hwthread-cpus` to the mpi call.
        executable (str): The LAMMPS executable to use.
        working_directory (str): Where the calculation will be executed (the ASE
            `tmp_dir` parameter). Defaults to `"."`, the current working directory.

    Returns:
        DelayedCalculator: A middle-man class whose :meth:`instantiate` method will
            create a :class:`ase.calculators.lammpsrun.LAMMPS` instance.

    Raises:
        ValueError: When any inconsistency is detected in the model definition, e.g. if
            model definitons are provided by both :param:`species` and
            :param:`ase_kwargs`.
    """
    ase_kwargs = {} if ase_kwargs is None else ase_kwargs

    if potential_parameters is not None:
        key_overlap = LammpsPotentialParameters.key_overlap(ase_kwargs)
        if len(key_overlap) > 0:
            raise ValueError(
                f"Found keys {key_overlap} in `ase_kwargs` and `potential_parameters`"
                f" was not None. These must not both be given, and if they are "
                f"provided in `ase_params`, all of {LammpsPotentialParameters.key_set} "
                f"must be given."
            )
        ase_kwargs.update(potential_parameters.asdict())

    if "command" in ase_kwargs.keys():
        raise ValueError(
            f"Specify the run command using the node-level input instead of the ase "
            f"keyword argument dictionary directly."
        )
    else:
        ase_kwargs["command"] = _maybe_mpi(
            mpi_command, n_cores, executable, oversubscribe, use_hwthread_cpus
        )

    ase_kwargs["tmp_dir"] = working_directory

    return DelayedLammps(cls_kwargs=ase_kwargs)
