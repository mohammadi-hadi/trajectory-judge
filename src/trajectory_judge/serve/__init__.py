"""The HTTP service around the judges.

Installed with the ``serve`` extra. Nothing here is imported by the library, and the package
root deliberately does not re-export it, so a plain install stays a three-dependency package.
"""

from __future__ import annotations

from trajectory_judge.serve.app import create_app

__all__ = ["create_app"]
