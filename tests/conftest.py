from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def examples() -> Path:
    return EXAMPLES


def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests unless a device is actually present."""
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="no GPU available")
    for item in items:
        if "gpu" in item.keywords or "amd" in item.keywords or "nvidia" in item.keywords:
            item.add_marker(skip)
