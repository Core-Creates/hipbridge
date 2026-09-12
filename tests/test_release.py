"""Publishing is the one thing here that cannot be taken back.

A version on PyPI is permanent. It can be yanked and never replaced, so an
artifact published under the wrong number is wrong for good. These tests are
cheap and they run on every push, which is the only time they are useful:
afterwards is too late.

Nothing here needs torch or Triton. It has to pass on the core legs, which
install neither.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        import tomli as tomllib

    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _workflow(name: str) -> dict:
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_the_package_version_and_the_module_agree():
    """Two places name the version, and nothing compared them.

    `pip install hipbridge==0.1.0` followed by `hipbridge.__version__` reporting
    something else is the kind of discrepancy that is only ever found by a user,
    and only after it is published.
    """
    import hipbridge

    assert hipbridge.__version__ == _pyproject()["project"]["version"]


def test_the_version_is_a_version():
    """Guards the shape, so a typo is caught before a tag is cut."""
    version = _pyproject()["project"]["version"]

    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc|\.dev)\d+)?", version), version


def test_publishing_needs_a_tag_or_a_person():
    """Never on a push to a branch. The mistake is permanent."""
    triggers = _workflow("release.yml").get("on") or _workflow("release.yml").get(True)

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"] == {"tags": ["v*"]}, (
        f"release fires on {triggers['push']}, which would publish from a branch"
    )


def test_publishing_stores_no_credential():
    """Trusted publishing, not a token sitting in repository secrets.

    A stored PyPI token is standing authority to publish, readable by any
    workflow in the repository and good until somebody rotates it. The OIDC
    exchange mints one that uploads once and expires. This project spent a day
    removing a credential from a machine; it should not add one back here.
    """
    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    job = _workflow("release.yml")["jobs"]["publish"]

    assert "secrets." not in text, "release must not read a repository secret"
    assert (
        job.get("permissions", {}).get("id-token")
        or _workflow("release.yml").get("permissions", {}).get("id-token") == "write"
    ), "trusted publishing needs an id-token"
    assert "mint-token" in text, "the short-lived token has to be minted, not stored"


def test_the_environment_matches_the_index_being_published_to():
    """The OIDC assertion carries the environment as a claim.

    A trusted publisher registered with an environment will not match a token
    that carries none, and the index reports that as "no corresponding
    publisher" - which reads as a missing publisher rather than a disagreeing
    one, and sends you to the wrong page. The job had no environment while
    TestPyPI's publisher named one, and that is how the second rehearsal failed.

    Named for the target, because one job publishes to both indexes and each
    registers its own publisher.
    """
    job = _workflow("release.yml")["jobs"]["publish"]
    environment = job.get("environment")

    assert environment, (
        "the publisher on each index is registered with an environment, so a job "
        "without one mints a token whose claims cannot match"
    )
    assert "inputs.repository" in str(environment), (
        f"environment is {environment!r}, which cannot be right for both indexes: "
        f"a rehearsal publishes to testpypi and a tag to pypi"
    )


def test_a_tag_cannot_publish_a_different_version():
    """v0.2.0 built from a tree that says 0.1.0 publishes one under the other."""
    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "GITHUB_REF_NAME" in text and "pyproject.toml" in text, (
        "nothing compares the tag against the version it would publish"
    )
    assert "twine check" in text, "a malformed artifact should not become permanent"


@pytest.mark.parametrize("name", ["release.yml"])
def test_release_actions_are_pinned_to_commits(name: str):
    """The repository requires it, and a release is where it matters most."""
    text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")

    for ref in re.findall(r"uses: (\S+)", text):
        _, _, version = ref.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", version), f"{ref} is not pinned to a commit"
