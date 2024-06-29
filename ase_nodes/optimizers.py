from __future__ import annotations

from ase.optimize import BFGS as ASEBFGS
from ase.optimize.optimize import Optimizer
from pyiron_workflow import as_function_node

from ase_nodes.shared import DelayedInstantiator


class DelayedOptimizer(DelayedInstantiator):
    pass
DelayedOptimizer.__init__.__annotations__.update({"cls": type[Optimizer]})
DelayedOptimizer.instantiate.__annotations__.update({"return": Optimizer})


@as_function_node("optimizer")
def BFGS() -> DelayedOptimizer:
    # This is a prototype -- don't bother exposing any optimizer inputs yet
    return DelayedOptimizer(ASEBFGS)
