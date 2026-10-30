from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "spec0_constraints.py"
_spec = importlib.util.spec_from_file_location("spec0_constraints", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
sys.modules.setdefault("spec0_constraints", _mod)

load_schedule = _mod.load_schedule
resolve_versions = _mod.resolve_versions
normalise_version = _mod.normalise_version
write_constraints = _mod.write_constraints
main = _mod.main


# Sparse synthetic schedule mirroring the real format; covers Q1-Q4 2025
# plus one future 2026 entry to verify exclusion logic.
SYNTHETIC_SCHEDULE = [
    {
        "start_date": "2025-01-01T00:00:00Z",
        "packages": {"numpy": "1.26.0", "pandas": "2.1.0", "scipy": "1.11.0"},
    },
    {
        "start_date": "2025-04-01T00:00:00Z",
        "packages": {"numpy": "2.0.0", "scikit-learn": "1.4.0"},
    },
    {
        "start_date": "2025-07-01T00:00:00Z",
        "packages": {"matplotlib": "3.9.0", "pandas": "2.2.0", "scipy": "1.13.0"},
    },
    {
        "start_date": "2025-10-01T00:00:00Z",
        "packages": {"numpy": "2.1.0", "scikit-learn": "1.5.0"},
    },
    {
        "start_date": "2026-01-01T00:00:00Z",
        "packages": {"numpy": "2.2.0", "pandas": "2.3.0"},
    },
]


@pytest.fixture()
def schedule_file(tmp_path: Path) -> Path:
    p = tmp_path / "schedule.json"
    p.write_text(json.dumps(SYNTHETIC_SCHEDULE), encoding="utf-8")
    return p


class TestNormaliseVersion:
    def test_three_component_unchanged(self) -> None:
        assert normalise_version("2.0.0") == "2.0.0"

    def test_two_component_gets_zero(self) -> None:
        assert normalise_version("3.4") == "3.4.0"

    def test_four_component_unchanged(self) -> None:
        assert normalise_version("1.2.3.4") == "1.2.3.4"

    def test_single_component_unchanged(self) -> None:
        assert normalise_version("2") == "2"


class TestLoadSchedule:
    def test_loads_local_file(self, schedule_file: Path) -> None:
        result = load_schedule(str(schedule_file), date(2025, 10, 1))
        assert isinstance(result, list)
        assert len(result) == len(SYNTHETIC_SCHEDULE)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_schedule(str(tmp_path / "nonexistent.json"), date(2025, 1, 1))

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_schedule(str(bad), date(2025, 1, 1))

    def test_non_array_json_raises(self, tmp_path: Path) -> None:
        obj = tmp_path / "obj.json"
        obj.write_text('{"numpy": "2.0"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_schedule(str(obj), date(2025, 1, 1))


class TestResolveVersions:
    def test_all_packages_resolved(self) -> None:
        versions, missing = resolve_versions(
            SYNTHETIC_SCHEDULE,
            ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"],
            date(2025, 10, 15),
        )
        assert missing == []
        assert versions["numpy"] == "2.1.0"
        assert versions["scipy"] == "1.13.0"
        assert versions["pandas"] == "2.2.0"
        assert versions["scikit-learn"] == "1.5.0"
        assert versions["matplotlib"] == "3.9.0"

    def test_future_entries_excluded(self) -> None:
        versions, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy", "pandas"], date(2025, 12, 31))
        assert missing == []
        assert versions["numpy"] == "2.1.0"  # 2026 entry (2.2.0) must be ignored
        assert versions["pandas"] == "2.2.0"

    def test_entry_on_exact_ref_date_included(self) -> None:
        versions, _ = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy"], date(2025, 10, 1))
        assert versions["numpy"] == "2.1.0"

    def test_later_entry_wins(self) -> None:
        versions, _ = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy"], date(2025, 12, 1))
        assert versions["numpy"] == "2.1.0"  # not Q2's 2.0.0

    def test_earliest_ref_date_only_sees_first_entry(self) -> None:
        versions, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy", "pandas", "scipy"], date(2025, 1, 1))
        assert missing == []
        assert versions["numpy"] == "1.26.0"
        assert versions["pandas"] == "2.1.0"
        assert versions["scipy"] == "1.11.0"

    def test_package_subset(self) -> None:
        versions, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["scipy"], date(2025, 8, 1))
        assert missing == []
        assert versions["scipy"] == "1.13.0"

    def test_unknown_package_goes_to_missing(self) -> None:
        versions, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy", "totally-unknown-pkg"], date(2025, 12, 1))
        assert "totally-unknown-pkg" in missing
        assert "numpy" in versions

    def test_package_resolved_from_earlier_entry_not_missing(self) -> None:
        versions, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["pandas"], date(2025, 12, 31))
        assert "pandas" not in missing
        assert versions["pandas"] == "2.2.0"

    def test_empty_schedule_returns_all_missing(self) -> None:
        versions, missing = resolve_versions([], ["numpy", "scipy"], date(2025, 6, 1))
        assert versions == {}
        assert set(missing) == {"numpy", "scipy"}

    def test_ref_date_before_all_entries_returns_missing(self) -> None:
        _, missing = resolve_versions(SYNTHETIC_SCHEDULE, ["numpy"], date(2024, 12, 31))
        assert "numpy" in missing

    def test_missing_start_date_raises(self) -> None:
        bad = [{"packages": {"numpy": "2.0.0"}}]
        with pytest.raises(ValueError, match="start_date"):
            resolve_versions(bad, ["numpy"], date(2025, 6, 1))

    def test_invalid_start_date_format_raises(self) -> None:
        bad = [{"start_date": "not-a-date", "packages": {"numpy": "2.0.0"}}]
        with pytest.raises(ValueError, match="parse start_date"):
            resolve_versions(bad, ["numpy"], date(2025, 6, 1))

    def test_packages_not_a_dict_raises(self) -> None:
        bad = [{"start_date": "2025-01-01T00:00:00Z", "packages": ["numpy"]}]
        with pytest.raises(ValueError, match="invalid 'packages'"):
            resolve_versions(bad, ["numpy"], date(2025, 6, 1))

    def test_version_not_a_string_raises(self) -> None:
        bad = [{"start_date": "2025-01-01T00:00:00Z", "packages": {"numpy": 200}}]
        with pytest.raises(ValueError, match="not a string"):
            resolve_versions(bad, ["numpy"], date(2025, 6, 1))


class TestWriteConstraints:
    def test_exact_pins_present(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"numpy": "2.1.0", "scipy": "1.13.0", "pandas": "2.2.0"}, str(out))
        content = out.read_text(encoding="utf-8")
        assert "numpy==2.1.0" in content
        assert "scipy==1.13.0" in content
        assert "pandas==2.2.0" in content

    def test_alphabetically_sorted(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"scipy": "1.13.0", "numpy": "2.1.0", "matplotlib": "3.9.0"}, str(out))
        lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if "==" in ln]
        assert lines == sorted(lines)

    def test_two_component_version_normalised(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"networkx": "3.4"}, str(out))
        assert "networkx==3.4.0" in out.read_text(encoding="utf-8")

    def test_no_greater_than_or_equal(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"numpy": "2.0.0"}, str(out))
        content = out.read_text(encoding="utf-8")
        assert ">=" not in content
        assert "numpy==2.0.0" in content

    def test_ends_with_newline(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"numpy": "2.0.0"}, str(out))
        assert out.read_text(encoding="utf-8").endswith("\n")

    def test_header_comment_present(self, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        write_constraints({"numpy": "2.0.0"}, str(out))
        assert out.read_text(encoding="utf-8").startswith("#")


class TestMain:
    def test_success(self, schedule_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        rc = main(
            [
                "--packages",
                "numpy",
                "scipy",
                "--output",
                str(out),
                "--date",
                "2025-10-15",
                "--schedule",
                str(schedule_file),
            ]
        )
        assert rc == 0
        content = out.read_text(encoding="utf-8")
        assert "numpy==2.1.0" in content
        assert "scipy==1.13.0" in content

    def test_json_output(self, schedule_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "constraints.txt"
        json_out = tmp_path / "versions.json"
        rc = main(
            [
                "--packages",
                "numpy",
                "--output",
                str(out),
                "--date",
                "2025-10-15",
                "--schedule",
                str(schedule_file),
                "--json-output",
                str(json_out),
            ]
        )
        assert rc == 0
        assert json.loads(json_out.read_text(encoding="utf-8")) == {"numpy": "2.1.0"}

    def test_exit_1_on_missing_package(self, schedule_file: Path, tmp_path: Path) -> None:
        rc = main(
            [
                "--packages",
                "numpy",
                "nonexistent-pkg",
                "--output",
                str(tmp_path / "c.txt"),
                "--date",
                "2025-10-15",
                "--schedule",
                str(schedule_file),
            ]
        )
        assert rc == 1

    def test_exit_2_on_bad_date(self, schedule_file: Path, tmp_path: Path) -> None:
        rc = main(
            [
                "--packages",
                "numpy",
                "--output",
                str(tmp_path / "c.txt"),
                "--date",
                "not-a-date",
                "--schedule",
                str(schedule_file),
            ]
        )
        assert rc == 2

    def test_exit_2_on_missing_schedule(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--packages",
                "numpy",
                "--output",
                str(tmp_path / "c.txt"),
                "--date",
                "2025-10-15",
                "--schedule",
                str(tmp_path / "no_such_file.json"),
            ]
        )
        assert rc == 2

    def test_exit_2_on_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        rc = main(
            [
                "--packages",
                "numpy",
                "--output",
                str(tmp_path / "c.txt"),
                "--date",
                "2025-10-15",
                "--schedule",
                str(bad),
            ]
        )
        assert rc == 2

    def test_github_output_written(self, schedule_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        gh_out = tmp_path / "github_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
        out = tmp_path / "constraints.txt"
        rc = main(
            [
                "--packages",
                "numpy",
                "--output",
                str(out),
                "--date",
                "2025-10-15",
                "--schedule",
                str(schedule_file),
            ]
        )
        assert rc == 0
        content = gh_out.read_text(encoding="utf-8")
        assert f"constraints-file={out}" in content
        assert "versions-json=" in content
        assert '"numpy"' in content

    def test_default_packages(self, tmp_path: Path) -> None:
        schedule = [
            {
                "start_date": "2025-01-01T00:00:00Z",
                "packages": {
                    "numpy": "2.0.0",
                    "scipy": "1.11.0",
                    "pandas": "2.1.0",
                    "scikit-learn": "1.4.0",
                    "matplotlib": "3.8.0",
                },
            }
        ]
        sf = tmp_path / "schedule.json"
        sf.write_text(json.dumps(schedule), encoding="utf-8")
        out = tmp_path / "constraints.txt"
        rc = main(["--output", str(out), "--date", "2025-06-01", "--schedule", str(sf)])
        assert rc == 0
        content = out.read_text(encoding="utf-8")
        for pkg in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]:
            assert pkg in content

    def test_accumulation_across_entries(self, tmp_path: Path) -> None:
        schedule = [
            {
                "start_date": "2024-01-01T00:00:00Z",
                "packages": {"numpy": "1.24.0", "scipy": "1.10.0"},
            },
            {"start_date": "2024-04-01T00:00:00Z", "packages": {"numpy": "1.26.0"}},
            {
                "start_date": "2024-10-01T00:00:00Z",
                "packages": {"numpy": "2.0.0", "scipy": "1.11.0"},
            },
            {
                "start_date": "2025-01-01T00:00:00Z",
                "packages": {"numpy": "2.1.0"},
            },  # future
        ]
        sf = tmp_path / "schedule.json"
        sf.write_text(json.dumps(schedule), encoding="utf-8")
        out = tmp_path / "constraints.txt"
        rc = main(
            [
                "--packages",
                "numpy",
                "scipy",
                "--output",
                str(out),
                "--date",
                "2024-12-31",
                "--schedule",
                str(sf),
            ]
        )
        assert rc == 0
        content = out.read_text(encoding="utf-8")
        assert "numpy==2.0.0" in content  # Q4 2024, not the future 2025 entry
        assert "scipy==1.11.0" in content

    def test_no_file_written_on_failure(self, tmp_path: Path) -> None:
        schedule = [{"start_date": "2025-01-01T00:00:00Z", "packages": {"numpy": "2.0.0"}}]
        sf = tmp_path / "schedule.json"
        sf.write_text(json.dumps(schedule), encoding="utf-8")
        out = tmp_path / "constraints.txt"
        rc = main(
            [
                "--packages",
                "totally-unknown",
                "--output",
                str(out),
                "--date",
                "2025-06-01",
                "--schedule",
                str(sf),
            ]
        )
        assert rc == 1
        assert not out.exists()

    def test_real_schedule_format(self, tmp_path: Path) -> None:
        """Smoke-test against data matching the actual Q4 2026 schedule.json."""
        real_format = [
            {
                "packages": {
                    "pandas": "2.3.0",
                    "scikit-learn": "1.5.0",
                    "scipy": "1.13.0",
                },
                "start_date": "2026-01-01T00:00:00Z",
            },
            {
                "packages": {
                    "matplotlib": "3.10.0",
                    "numpy": "2.1.0",
                    "scikit-learn": "1.6.0",
                    "scipy": "1.15.0",
                },
                "start_date": "2026-04-01T00:00:00Z",
            },
            {"packages": {"numpy": "2.2.0"}, "start_date": "2026-07-01T00:00:00Z"},
            {
                "packages": {
                    "matplotlib": "3.11.0",
                    "numpy": "2.3.0",
                    "scikit-learn": "1.7.0",
                },
                "start_date": "2026-10-01T00:00:00Z",
            },
        ]
        sf = tmp_path / "schedule.json"
        sf.write_text(json.dumps(real_format), encoding="utf-8")
        out = tmp_path / "constraints.txt"
        rc = main(
            [
                "--packages",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "--output",
                str(out),
                "--date",
                "2026-10-30",
                "--schedule",
                str(sf),
            ]
        )
        assert rc == 0
        content = out.read_text(encoding="utf-8")
        assert "numpy==2.3.0" in content
        assert "scipy==1.15.0" in content
        assert "pandas==2.3.0" in content
        assert "scikit-learn==1.7.0" in content
        assert "matplotlib==3.11.0" in content
