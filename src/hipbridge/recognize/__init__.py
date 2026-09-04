"""Kernel shape recognition. No default pattern, no catch-all."""

from hipbridge.recognize import rules  # noqa: F401  (import registers the rules)
from hipbridge.recognize.base import recognize, registered, rule

__all__ = ["recognize", "registered", "rule"]
