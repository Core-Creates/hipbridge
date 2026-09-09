"""Kernel shape recognition. No default pattern, no catch-all."""

from hipbridge.recognizers import rules  # noqa: F401  (import registers the rules)
from hipbridge.recognizers.base import recognize, registered, rule

__all__ = ["recognize", "registered", "rule"]
