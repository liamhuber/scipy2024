This is a collection of `pyiron_workflow` nodes wrapping a small subset of `ase` functionality.
Since `ase` leans heavily on OOP, e.g. running calculations by setting the `calculator` attribute of an atomic structure, we jump through a couple of hoops to bend it into a functional and idempotent paradigm.
In particular, we introduce classes which serve as instantiation wrappers for key objects (calculators, filters, optimizers...) allowing us to use those object in a "delayed" way.
This is also very useful for the file-based calculators, which begin writing files to some directory at their instantiation time -- perhaps before we've even decided what directory they should do that in!

This node package is built-to-purpose for the scipy2024 use case example at the main level of the repo, and is neither exhaustive nor thoroughly tested beyond that use case.
However, it is hopefully still pedagogically useful for how to wrap existing python tools as a node package.
If this wrapping were to be continued, one can imagine a few relatively straightforward abstractions that should be implemented:
- Many routines that work with an optimizer can work with other evolvers; this interface should be generalized
- Many routines take collections of the same input; the philosophy of the dataclass introduced for minimization input could be expanded upon (and `pyiron_workflow` dataclass nodes leveraged where useful)
- Adapters should be layered on top of the existing calculators to use universal language for the various code-specific parameters (where different codes have parameters with the same physical meaning but different names or units)