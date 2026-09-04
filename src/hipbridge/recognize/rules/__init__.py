"""Importing this package registers every rule with the registry."""

from hipbridge.recognize.rules import elementwise, reductions  # noqa: F401

__all__ = ["elementwise", "reductions"]
