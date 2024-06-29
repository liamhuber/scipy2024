"""
Nodes wrapping functionality from the `ase` library.

This node package is built-to-purpose for the accompanying solute-vacancy binding
energy example; it does not give access to all functionality in `ase`, nor even all
input and configuration parameters for the objects exposed, nor is it extensively
optimized for performance. Rather, it is intended for educational purposes, to show
how other (even complex and powerful) tools can be packaged and used in a workflow
formulation.

To coerce `ase` into a functional paradigm, the main attack was to delay the
instantiation of many objects until the last possible moment prior to a calculation.
"""

# All public nodes and functions in the following submodules represent the API
import ase_nodes.atoms
import ase_nodes.calculators
import ase_nodes.filters
import ase_nodes.optimizers
import ase_nodes.physics
