"""The README's quickstart is executed, and its documented output is asserted.

The repository's rule is that no number appears in the documentation unless
somebody measured it. A rule nobody checks is a rule that decays, so this test
extracts the quickstart from `README.md`, runs it in a subprocess, and compares
what it printed against the block the README says it prints.

A subprocess rather than `exec`: the quickstart is meant to work as a file a
reader saves and runs, and running it in this process would let it pass because
of something the test suite had already imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
README = REPOSITORY / "README.md"


def blocks(language: str) -> list[str]:
    """Every fenced code block in the README with the given info string.

    Scanned line by line rather than with one regular expression. A non-greedy
    pattern pairs an opening fence with the nearest following fence, which in a
    document containing both tagged and untagged blocks pairs a *closing* fence
    with the next *opening* one and silently returns the prose in between.
    """
    found: list[str] = []
    collecting: list[str] | None = None
    skipping = False
    for line in README.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("```"):
            info = line.strip()[3:]
            if collecting is not None:
                found.append("".join(collecting))
                collecting = None
            elif skipping:
                skipping = False
            elif info == language:
                collecting = []
            else:
                skipping = True
        elif collecting is not None:
            collecting.append(line)
    return found


def test_the_readme_quickstart_runs_and_prints_what_it_claims(tmp_path: Path) -> None:
    python_blocks = blocks("python")
    assert python_blocks, "the README no longer contains a python block"
    script = tmp_path / "quickstart.py"
    script.write_text(python_blocks[0], encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr

    # The first bare block after the quickstart is its documented output.
    documented = next(b for b in blocks("") if "accuracy after 20 epochs" in b)
    assert completed.stdout == documented, (
        f"the README's quickstart output is stale.\ndocumented:\n{documented}\n"
        f"actual:\n{completed.stdout}"
    )


@pytest.mark.parametrize(
    "example",
    ["examples/mlp_moons", "examples/mnist_style", "examples/char_transformer"],
)
def test_every_example_has_a_readme_with_its_measured_output(example: str) -> None:
    """Each example's README must exist and contain a transcript, not a promise."""
    readme = REPOSITORY / example / "README.md"
    assert readme.is_file(), f"{example} has no README"
    text = readme.read_text(encoding="utf-8")
    assert "## Measured output" in text
    assert "```" in text
    assert (REPOSITORY / example / "train.py").is_file()
