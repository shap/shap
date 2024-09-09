# Utility script to print dependency pins representing lowest supported versions
# Run with `uv run scripts/extract_lower_bounds.py`
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "packaging",
#     "tomli",
# ]
# ///

import tomli
from packaging.requirements import Requirement

EXTRA_CONSTRAINTS = [
    "Pillow==9.0.1",  # For compatibility with older numpy
]


def parse_lower_bounds(dependencies: list[str]) -> dict[str, str]:
    """Extract any declared ">=" lower bounds from a list of dependencies."""
    lower_bounds = {}
    for dep in dependencies:
        req = Requirement(dep)
        for spec in req.specifier:
            if spec.operator == ">=":
                lower_bounds[req.name] = spec.version
    return lower_bounds


def main():
    # Parse all declared lower bound dependencies from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        data = tomli.load(f)

    # Core dependencies
    bounds = parse_lower_bounds(data["project"]["dependencies"])
    # Optional dependencies
    optional_deps = data["project"]["optional-dependencies"]
    for group in ["test", "plots"]:
        bounds.update(parse_lower_bounds(optional_deps[group]))

    # Print out these dependencies as a list of pinned versions
    for dep, version in bounds.items():
        print(f"{dep}=={version}")

    # Some overrides
    for dep in EXTRA_CONSTRAINTS:
        print(dep)


if __name__ == "__main__":
    main()
