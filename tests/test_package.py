"""The packaging surface: what an installed distribution must expose.

Everything here runs identically under an editable install and against a built wheel,
which is what lets CI run one suite on both. Each test guards a regression the rest of
the suite cannot see: the module version drifting from pyproject, the PEP 561 marker
falling out of the wheel, or the console script pointing at a function that moved.
"""

from __future__ import annotations

from importlib.metadata import entry_points, version
from importlib.resources import files

import trajectory_judge

DIST = "trajectory-judge"


def test_module_version_matches_the_installed_metadata() -> None:
    assert trajectory_judge.__version__ == version(DIST)


def test_py_typed_ships_with_the_package() -> None:
    """The annotations are strict-mypy checked, but downstream type checkers only look
    at them if the marker file actually lands inside the wheel."""
    assert files("trajectory_judge").joinpath("py.typed").is_file()


def test_the_console_script_points_at_the_cli() -> None:
    (script,) = entry_points(group="console_scripts", name=DIST)
    assert script.value == "trajectory_judge.cli:app"


def test_every_name_in_all_is_importable_from_the_root() -> None:
    for name in trajectory_judge.__all__:
        assert hasattr(trajectory_judge, name), name
