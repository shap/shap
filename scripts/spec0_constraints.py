from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

SCHEDULE_URL_TEMPLATE = "https://github.com/scientific-python/spec0-action/releases/download/{tag}/schedule.json"

DEFAULT_PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]


def _quarter_tag(ref: date) -> str:
    quarter = (ref.month - 1) // 3 + 1
    return f"schedule-{ref.year}-Q{quarter}"


def load_schedule(schedule_path: str | None, ref_date: date) -> list[dict[str, Any]]:
    if schedule_path:
        p = Path(schedule_path)
        if not p.is_file():
            raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
        text = p.read_text(encoding="utf-8")
    else:
        tag = _quarter_tag(ref_date)
        url = SCHEDULE_URL_TEMPLATE.format(tag=tag)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                text = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to download SPEC 0 schedule from {url}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in schedule: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected schedule.json to be a JSON array, got {type(data).__name__!r}")
    return data  # type: ignore[return-value]


def resolve_versions(
    schedule: list[dict[str, Any]],
    packages: list[str],
    ref_date: date,
) -> tuple[dict[str, str], list[str]]:
    """Walk applicable schedule entries and return the latest floor per package.

    The schedule is sparse: each entry only lists packages that changed that
    quarter. We sort by date and let later entries overwrite earlier ones.
    Returns (versions, missing) where missing lists unresolved packages.
    """
    ref_dt = datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=UTC)
    requested = set(packages)
    versions: dict[str, str] = {}

    def _entry_dt(entry: dict[str, Any]) -> datetime:
        raw = entry.get("start_date")
        if not isinstance(raw, str):
            raise ValueError(f"Schedule entry is missing a 'start_date' string: {entry!r}")
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Cannot parse start_date {raw!r}: {exc}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    applicable = [e for e in schedule if _entry_dt(e) <= ref_dt]
    applicable.sort(key=_entry_dt)

    for entry in applicable:
        pkgs = entry.get("packages")
        if not isinstance(pkgs, dict):
            raw = entry.get("start_date", "<unknown>")
            raise ValueError(f"Schedule entry {raw!r} has invalid 'packages' field: {pkgs!r}")
        for pkg, ver in pkgs.items():
            if pkg not in requested:
                continue
            if not isinstance(ver, str):
                raw = entry.get("start_date", "<unknown>")
                raise ValueError(f"Version for {pkg!r} in {raw!r} is not a string: {ver!r}")
            versions[pkg] = ver

    missing = [p for p in packages if p not in versions]
    return versions, missing


def normalise_version(ver: str) -> str:
    """Pad a two-component version (X.Y) to X.Y.0; leave others unchanged."""
    parts = ver.split(".")
    if len(parts) == 2:  # noqa: PLR2004
        return f"{ver}.0"
    return ver


def write_constraints(versions: dict[str, str], output_path: str) -> None:
    lines = [
        "# SPEC 0 minimum dependency constraints — do not edit by hand.",
        "# Regenerate with: python scripts/spec0_constraints.py",
        "",
    ]
    for pkg in sorted(versions):
        lines.append(f"{pkg}=={normalise_version(versions[pkg])}")
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912
    parser = argparse.ArgumentParser(
        prog="spec0_constraints",
        description="Generate SPEC 0 minimum dependency constraints for CI.",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=DEFAULT_PACKAGES,
        metavar="PKG",
        help=f"Packages to pin (default: {' '.join(DEFAULT_PACKAGES)})",
    )
    parser.add_argument(
        "--output",
        default="constraints-spec0.txt",
        metavar="FILE",
        help="Output path for the constraints file (default: %(default)s)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Reference date; defaults to today. Use a fixed date for reproducible runs.",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        metavar="FILE",
        help="Local schedule.json path; skips network download.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        metavar="FILE",
        help="Also write the package→version mapping as JSON.",
    )
    args = parser.parse_args(argv)

    if args.date is not None:
        try:
            ref_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"ERROR: --date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
            return 2
    else:
        ref_date = date.today()

    print(f"[spec0] Reference date : {ref_date}", file=sys.stderr)
    print(f"[spec0] Packages       : {', '.join(args.packages)}", file=sys.stderr)

    try:
        schedule = load_schedule(args.schedule, ref_date)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: Invalid schedule data: {exc}", file=sys.stderr)
        return 2

    print(f"[spec0] Schedule entries loaded: {len(schedule)}", file=sys.stderr)

    try:
        versions, missing = resolve_versions(schedule, args.packages, ref_date)
    except ValueError as exc:
        print(f"ERROR: Invalid schedule data: {exc}", file=sys.stderr)
        return 2

    if missing:
        print(
            "ERROR: No SPEC 0 floor found for: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "       Package may not be tracked by SPEC 0, or the schedule does not cover this date.",
            file=sys.stderr,
        )
        return 1

    print("[spec0] Resolved floors:", file=sys.stderr)
    for pkg in sorted(versions):
        print(f"  {pkg}=={normalise_version(versions[pkg])}", file=sys.stderr)

    try:
        write_constraints(versions, args.output)
    except OSError as exc:
        print(f"ERROR: Cannot write {args.output!r}: {exc}", file=sys.stderr)
        return 2

    print(f"[spec0] Constraints written to: {args.output}", file=sys.stderr)

    versions_map = {pkg: normalise_version(v) for pkg, v in versions.items()}
    versions_json_str = json.dumps(versions_map, sort_keys=True)

    if args.json_output:
        try:
            Path(args.json_output).write_text(versions_json_str, encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: Cannot write {args.json_output!r}: {exc}", file=sys.stderr)
            return 2
        print(f"[spec0] JSON mapping written to: {args.json_output}", file=sys.stderr)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        try:
            with open(github_output, "a", encoding="utf-8") as fh:
                fh.write(f"constraints-file={args.output}\n")
                fh.write(f"versions-json={versions_json_str}\n")
        except OSError as exc:
            print(
                f"ERROR: Cannot write GITHUB_OUTPUT {github_output!r}: {exc}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
