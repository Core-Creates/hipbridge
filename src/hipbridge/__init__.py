"""hipbridge: recognize CUDA kernels, substitute verified AMD implementations.

Core imports cleanly with libclang alone. The heavy pieces live behind extras
and are reached through hipbridge.kernels / hipbridge.verify, each of which exposes
available() so callers can degrade instead of crashing.
"""

from hipbridge.analysis import ARCHS, lds_padding, occupancy, roofline
from hipbridge.frontend import KernelFacts, Pattern, Recognition, parse_file, parse_source
from hipbridge.recognize import recognize, registered

__version__ = "0.1.0.dev0"

__all__ = [
    "ARCHS",
    "KernelFacts",
    "Pattern",
    "Recognition",
    "__version__",
    "lds_padding",
    "occupancy",
    "parse_file",
    "parse_source",
    "recognize",
    "registered",
    "roofline",
]
