from __future__ import annotations

from abc import ABC
from typing import Literal


class DelayedInstantiator(ABC):
    """
    It takes some massaging to get ASE to work in a functional way; in particular,
    many things are non-functional in that they are added onto an :class:`ase.Atoms`
    instance, and/or such an object is required at instantiation of an object (after
    which it is difficult to modify). Some objects, such as particular calculators,
    start writing files as soon as they're instantiated.

    This class is a workaround to that problem, where we delay instantiating many of
    these ASE objects until the last possible moment.
    """
    def __init__(self, cls: type, cls_args: tuple = (), cls_kwargs: dict | None = None):
        self.cls = cls
        self.cls_args = cls_args
        self.cls_kwargs = {} if cls_kwargs is None else cls_kwargs

    def instantiate(
        self,
        *args,
        args_behavior: Literal[
            "fail", "ignore_stored", "ignore_new", "first_new", "first_stored"
        ] = "fail",
        kwargs_behavior: Literal[
            "fail", "new_updates_stored", "stored_updates_new"
        ] = "new_updates_stored",
        **kwargs,
    ):
        """
        Instantiate the stored class, possibly overriding or augmentic stored args and
        kwargs (if any)

        Args:
            *args:
            args_behavior:
            kwargs_behavior:
            **kwargs:

        Returns:
            An instance of the stored class.

        Raises:
            ValueError: If (kw)args are both stored _and_ provided and the
                corresponding behavior parameter is set to fail.
        """
        if len(args) > 0 and len(self.cls_args) > 0:
            match args_behavior:
                case "fail":
                    raise ValueError(
                        f"{self.__class__.__name__}.make trying to make {self.cls} got "
                        f"args {args} when it had stored args {self.cls_args} and was"
                        f" set to fail on overlap."
                    )
                case "ignore_stored":
                    pass
                case "ignore_new":
                    args = self.cls_args
                case "first_new":
                    args = args + self.cls_args
                case "first_stored":
                    args = self.cls_args + args
        elif len(self.cls_args) > 0:
            args = self.cls_args

        if len(kwargs) > 0 and len(self.cls_kwargs) > 0:
            match kwargs_behavior:
                case "fail":
                    raise ValueError(
                        f"{self.__class__.__name__}.make trying to make {self.cls} got "
                        f"kwargs {kwargs} when it had stored kwargs {self.cls_kwargs} "
                        f"and was set to fail on overlap."
                    )
                case "new_updates_stored":
                    stored = dict(self.cls_kwargs)
                    stored.update(kwargs)
                    kwargs = stored
                case "stored_updates_new":
                    kwargs.update(self.cls_kwargs)
        elif len(self.cls_kwargs) > 0:
            kwargs = self.cls_kwargs

        return self.cls(*args, **kwargs)
