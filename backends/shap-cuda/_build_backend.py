"""PEP 517 build backend wrapping scikit-build-core.

shap_cuda/cext/tree_shap.h is a symlink into the sibling shap/cext/ dir (the
CPU and GPU extensions compile against the same header). scikit-build-core's
sdist packaging preserves symlinks as-is, which produces a tarball that's
broken once extracted outside this monorepo. build_sdist here swaps the
symlink for its resolved file content before delegating, then restores it.
"""

import shutil
from pathlib import Path
from typing import Any

from scikit_build_core import build as _skbuild

_SYMLINKED_HEADERS: list[Path] = [Path(__file__).parent / "shap_cuda" / "cext" / "tree_shap.h"]


def __getattr__(name: str) -> Any:
    # Every PEP 517 hook other than build_sdist delegates to scikit-build-core unmodified.
    return getattr(_skbuild, name)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    materialized: list[tuple[Path, Path]] = []
    try:
        for header in _SYMLINKED_HEADERS:
            if header.is_symlink():
                resolved = header.resolve()
                relative_target = Path("../../../../shap/cext") / header.name
                header.unlink()
                shutil.copyfile(resolved, header)
                materialized.append((header, relative_target))
        return _skbuild.build_sdist(sdist_directory, config_settings)
    finally:
        for header, relative_target in materialized:
            header.unlink()
            header.symlink_to(relative_target)
