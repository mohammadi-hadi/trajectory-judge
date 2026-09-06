"""Assert a benchmark run is sane. Not a performance gate.

The ceilings here sit roughly an order of magnitude above what a laptop observes, because a
shared CI runner cannot measure a 15% regression honestly. What this does catch is a blocking
call landing in the request path, a scenario that starts erroring, or a service overhead that
has grown from microseconds to something a user would feel.

    python bench/check_invariants.py bench/results/overhead.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

#: Scenarios that are supposed to produce failures, because that is what they measure.
EXPECT_ERRORS = {"degraded", "overload"}
NO_UPSTREAM = {"floor", "rules"}

MAX_OVERHEAD_P99_MS = 250.0
MAX_NO_UPSTREAM_P99_MS = 2000.0


def problems(data: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for scenario in data["scenarios"]:
        tag = f"{scenario['name']}@c{scenario['concurrency']}"
        if scenario["name"] not in EXPECT_ERRORS and scenario["error_rate"] > 0:
            found.append(f"{tag}: error rate {scenario['error_rate']} where none was expected")
        if scenario["name"] in NO_UPSTREAM and scenario["p99_ms"] > MAX_NO_UPSTREAM_P99_MS:
            found.append(f"{tag}: p99 {scenario['p99_ms']:.0f}ms with no upstream at all")
        overhead = scenario["overhead_p99_ms"]
        if not math.isnan(overhead) and overhead > MAX_OVERHEAD_P99_MS:
            found.append(f"{tag}: service overhead p99 {overhead:.1f}ms")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()

    with open(args.path) as handle:
        data = json.load(handle)

    found = problems(data)
    if found:
        print("\n".join(found), file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(data['scenarios'])} scenarios within the invariants")


if __name__ == "__main__":
    main()
