from __future__ import annotations

from ase.filters import Filter, UnitCellFilter as ASEUnitCellFilter
from pyiron_workflow import as_function_node

from ase_nodes.shared import DelayedInstantiator


class DelayedFilter(DelayedInstantiator):
    pass
DelayedFilter.__init__.__annotations__.update({"cls": type[Filter]})
DelayedFilter.instantiate.__annotations__.update({"return": Filter})


@as_function_node("filter_")
def UnitCellFilter(scalar_pressure: float = 0.0) -> DelayedFilter:
    # This is a prototype -- only expose the single most interesting filter input
    return DelayedFilter(
        ASEUnitCellFilter,
        cls_kwargs={"scalar_pressure": scalar_pressure}
    )
