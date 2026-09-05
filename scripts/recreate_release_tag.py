"""Validate HEAD, then create or replace an annotated release tag."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from subprocess import run
from tomllib import loads

from .release_metadata import check_main, release_notes


def git(repository: Path, *arguments: str, check: bool = True) -> str:
    """Run git in the repository and return stripped stdout."""
    result = run(
        ["git", *arguments],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def recreate_release_tag(
    repository: Path,
    tag: str,
    *,
    remote: str = "origin",
    main_branch: str = "main",
    replace_existing: bool = False,
    dry_run: bool = False,
) -> str:
    """Push an annotated tag at HEAD after validating release metadata.

    Existing tags are left alone unless ``replace_existing`` is true. Validation
    and ``git fetch`` run before any delete so a failed changelog check cannot
    remove a remote tag.
    """
    project = loads((repository / "pyproject.toml").read_text())
    release_notes(
        tag,
        project["project"]["version"],
        (repository / "CHANGELOG.md").read_text(),
    )
    git(repository, "fetch", remote)
    if git(repository, "status", "--porcelain"):
        raise ValueError("working tree must be clean")
    commit = git(repository, "rev-parse", "HEAD")
    tip = git(repository, "rev-parse", f"{remote}/{main_branch}")
    if commit != tip:
        raise ValueError(f"HEAD must match {remote}/{main_branch} before tagging")
    check_main(repository, f"{remote}/{main_branch}")
    local_tag = git(repository, "tag", "--list", tag)
    remote_tag = git(repository, "ls-remote", "--tags", remote, f"refs/tags/{tag}")
    if (local_tag or remote_tag) and not replace_existing:
        raise ValueError(
            f"{tag} already exists; pass --replace-existing only after confirming "
            "nobody installed from that tag and no GitHub Release consumed it"
        )
    if dry_run:
        print(f"Would tag {tag} at {commit}")
        return commit
    if local_tag:
        git(repository, "tag", "--delete", tag)
    if remote_tag:
        git(repository, "push", remote, "--delete", tag)
    git(repository, "tag", "-a", tag, "-m", tag)
    git(repository, "push", remote, tag)
    print(f"Pushed {tag} at {commit}")
    return commit


def main() -> None:
    """Replace a failed release tag only after the same checks CI will run."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--main-branch", default="main")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="delete the local and remote tag before recreating it at HEAD",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recreate_release_tag(
        Path(__file__).resolve().parents[1],
        args.tag,
        remote=args.remote,
        main_branch=args.main_branch,
        replace_existing=args.replace_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
