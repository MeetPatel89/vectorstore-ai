from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tarfile import TarInfo
from tarfile import open as open_tar
from zipfile import ZipFile

import pytest

from scripts.check_dist import check_distributions

STEM = "vectorstore_ai-0.1.0"


def write_archives(
    directory: Path,
    *,
    typed: bool = True,
    license_name: str = "MIT",
    extra_source: str | None = None,
) -> tuple[Path, Path]:
    wheel = directory / f"{STEM}-py3-none-any.whl"
    sdist = directory / f"{STEM}.tar.gz"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("vectorstore/__init__.py", "")
        if typed:
            archive.writestr("vectorstore/py.typed", "")
        archive.writestr(f"{STEM}.dist-info/licenses/LICENSE", "MIT license text")
        archive.writestr(
            f"{STEM}.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: vectorstore-ai\nVersion: 0.1.0\n"
            f"License-Expression: {license_name}\n",
        )
    source = [
        "src/vectorstore/py.typed",
        "pyproject.toml",
        "LICENSE",
        "README.md",
        "main.py",
        "examples/sample_corpus/incident.md",
        "tests/package_smoke.py",
    ]
    if extra_source:
        source.append(extra_source)
    with open_tar(sdist, "w:gz") as archive:
        for name in source:
            archive.addfile(TarInfo(f"{STEM}/{name}"), BytesIO(b""))
    return wheel, sdist


def test_distribution_checks_record_exact_archive_checksums(tmp_path: Path) -> None:
    archives = write_archives(tmp_path)
    check_distributions(tmp_path, "0.1.0")
    assert (tmp_path / "SHA256SUMS").read_text() == "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(archives)
    )


@pytest.mark.parametrize(
    "source",
    [
        ".vscode/settings.json",
        ".cursor/rules/local.mdc",
        "data/private.md",
        "notebooks/demo.py",
        "examples/.env.local",
        "docs/retrieval.plan.md",
        "../outside",
    ],
)
def test_distribution_rejects_repository_only_content(
    tmp_path: Path, source: str
) -> None:
    write_archives(tmp_path, extra_source=source)
    with pytest.raises(ValueError, match="unexpected file"):
        check_distributions(tmp_path, "0.1.0")
    assert not (tmp_path / "SHA256SUMS").exists()


def test_distribution_requires_typing_marker_and_license(tmp_path: Path) -> None:
    write_archives(tmp_path, typed=False)
    with pytest.raises(ValueError, match="py.typed"):
        check_distributions(tmp_path, "0.1.0")
    write_archives(tmp_path, license_name="Apache-2.0")
    with pytest.raises(ValueError, match="MIT"):
        check_distributions(tmp_path, "0.1.0")


def test_distribution_rejects_stale_archives(tmp_path: Path) -> None:
    write_archives(tmp_path)
    (tmp_path / "vectorstore_ai-0.0.9.tar.gz").write_bytes(b"old")
    with pytest.raises(ValueError, match="exactly one"):
        check_distributions(tmp_path, "0.1.0")
