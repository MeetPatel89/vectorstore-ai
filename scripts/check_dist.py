"""Check distribution contents and metadata, then write their checksums."""

from __future__ import annotations

from argparse import ArgumentParser
from email.parser import BytesParser
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tarfile import open as open_tar
from tomllib import loads
from zipfile import ZipFile


def check_distributions(directory: Path, version: str) -> None:
    """Reject missing package files, wrong metadata, and repository-only content."""
    stem = f"vectorstore_ai-{version}"
    wheel = directory / f"{stem}-py3-none-any.whl"
    sdist = directory / f"{stem}.tar.gz"
    archives = set(directory.glob("*.whl")) | set(directory.glob("*.tar.gz"))
    if archives != {wheel, sdist}:
        raise ValueError(
            "expected exactly one wheel and one sdist at the project version"
        )
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {"vectorstore/__init__.py", "vectorstore/py.typed"}
        if not required <= names:
            raise ValueError("wheel is missing package source or py.typed")
        if any(
            not name.startswith(("vectorstore/", f"{stem}.dist-info/"))
            for name in names
        ):
            raise ValueError(
                "wheel contains files outside the package and its metadata"
            )
        metadata = BytesParser().parsebytes(archive.read(f"{stem}.dist-info/METADATA"))
        if metadata["Name"] != "vectorstore-ai" or metadata["Version"] != version:
            raise ValueError("wheel metadata does not match the project")
        if metadata["License-Expression"] != "MIT":
            raise ValueError("wheel must declare its MIT license")
        if f"{stem}.dist-info/licenses/LICENSE" not in names:
            raise ValueError("wheel is missing the license text")
    with open_tar(sdist) as archive:
        paths = {PurePosixPath(member.name) for member in archive.getmembers()}
        required_source = {
            "src/vectorstore/py.typed",
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "main.py",
            "examples/sample_corpus/incident.md",
            "tests/package_smoke.py",
        }
        if not {PurePosixPath(stem, name) for name in required_source} <= paths:
            raise ValueError(
                "sdist is missing required source, documentation, or examples"
            )
        forbidden = {
            ".git",
            ".vscode",
            ".cursor",
            ".venv",
            "__pycache__",
            "data",
            "notebooks",
        }
        for path in paths:
            if (
                path.is_absolute()
                or ".." in path.parts
                or forbidden.intersection(path.parts)
                or path.name.startswith(".env")
                or path.name.endswith(".plan.md")
            ):
                raise ValueError(f"sdist contains an unexpected file: {path}")
    checksums = "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted((wheel, sdist))
    )
    (directory / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def main() -> None:
    """Check the archives produced by the package job."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = loads((root / "pyproject.toml").read_text())["project"]["version"]
    check_distributions(args.directory, version)
    print(f"Validated vectorstore-ai {version} distributions")


if __name__ == "__main__":
    main()
