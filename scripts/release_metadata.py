"""Validate a stable release tag and extract its dated changelog entry."""

from __future__ import annotations

import re
from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from subprocess import run
from tomllib import loads

STABLE_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def release_notes(tag: str, version: str, changelog: str) -> str:
    """Return nonempty release notes after validating tag, version, and date."""
    if STABLE_TAG.fullmatch(tag) is None:
        raise ValueError("release tag must use stable vMAJOR.MINOR.PATCH syntax")
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match project version {version!r}")
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", re.MULTILINE
    )
    matches = list(heading.finditer(changelog))
    if len(matches) != 1:
        raise ValueError("changelog must contain exactly one dated release entry")
    match = matches[0]
    released_on = date.fromisoformat(match[1])
    if released_on > date.today():
        raise ValueError("changelog release date cannot be in the future")
    remainder = changelog[match.end() :]
    notes = re.split(r"^## ", remainder, maxsplit=1, flags=re.MULTILINE)[0].strip()
    if not notes or not any(
        line.strip().startswith("- ") for line in notes.splitlines()
    ):
        raise ValueError("release notes must contain at least one changelog bullet")
    return f"{match[0]}\n\n{notes}\n"


def check_main(repository: Path, main_ref: str = "origin/main") -> str:
    """Require the checked-out commit to be contained in the main branch."""
    commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = run(
        ["git", "merge-base", "--is-ancestor", commit, main_ref],
        cwd=repository,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"release commit must be contained in {main_ref}")
    return commit


def main() -> None:
    """Validate the current checkout before a tag can produce a release."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--notes-out", type=Path)
    parser.add_argument("--main-ref", default="origin/main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = loads((root / "pyproject.toml").read_text())
    notes = release_notes(
        args.tag, project["project"]["version"], (root / "CHANGELOG.md").read_text()
    )
    commit = check_main(root, args.main_ref)
    if args.notes_out is not None:
        args.notes_out.write_text(notes, encoding="utf-8")
    print(f"Validated {args.tag} at {commit}")


if __name__ == "__main__":
    main()
