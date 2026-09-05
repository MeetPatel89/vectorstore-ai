"""Publish already-validated assets without importing or installing the package."""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory
from typing import Any, cast

from .release_metadata import STABLE_TAG


class GitHubReleaseClient:
    """Use GitHub CLI authentication for release and asset operations."""

    def __init__(self, repository: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
            raise ValueError("repository must be OWNER/NAME")
        self.repository = repository

    def tag_commit(self, tag: str) -> str:
        """Resolve a remote tag to its commit, including annotated tags."""
        return (
            self._run(
                "api",
                f"repos/{self.repository}/commits/refs%2Ftags%2F{tag}",
                "--jq",
                ".sha",
            )
            .decode()
            .strip()
        )

    def release(self, tag: str) -> dict[str, Any] | None:
        """Read a release, distinguishing a missing release from API failures."""
        try:
            result = self._run("api", f"repos/{self.repository}/releases/tags/{tag}")
        except CalledProcessError as error:
            if b"HTTP 404" not in (error.stderr or b""):
                raise
            # REST's tag lookup does not find pending draft tags. Resolve the
            # draft through GraphQL, then fetch the same REST asset shape.
            owner, name = self.repository.split("/")
            release_id = (
                self._run(
                    "api",
                    "graphql",
                    "-f",
                    "query=query($owner:String!,$name:String!,$tag:String!) { "
                    "repository(owner:$owner,name:$name) { "
                    "release(tagName:$tag) { databaseId } } }",
                    "-F",
                    f"owner={owner}",
                    "-F",
                    f"name={name}",
                    "-F",
                    f"tag={tag}",
                    "--jq",
                    ".data.repository.release.databaseId",
                )
                .decode()
                .strip()
            )
            if release_id == "null":
                return None
            if not release_id.isdecimal():
                raise ValueError("GitHub returned an invalid draft release ID")
            result = self._run("api", f"repos/{self.repository}/releases/{release_id}")
        return cast(dict[str, Any], json.loads(result))

    def create_draft(self, tag: str, commit: str, notes: Path) -> None:
        """Create a draft only for an existing tag."""
        self._run(
            "release",
            "create",
            tag,
            "--repo",
            self.repository,
            "--draft",
            "--verify-tag",
            "--target",
            commit,
            "--title",
            tag,
            "--notes-file",
            str(notes),
        )

    def asset_bytes(self, asset_id: int) -> bytes:
        """Download an existing asset for comparison rather than trusting its name."""
        return self._run(
            "api",
            f"repos/{self.repository}/releases/assets/{asset_id}",
            "--header",
            "Accept: application/octet-stream",
        )

    def upload(self, tag: str, asset: Path) -> None:
        """Upload a new asset without replacing any existing asset."""
        self._run("release", "upload", tag, str(asset), "--repo", self.repository)

    def publish(self, tag: str) -> None:
        """Publish the fully populated draft."""
        self._run("release", "edit", tag, "--repo", self.repository, "--draft=false")

    @staticmethod
    def _run(*arguments: str) -> bytes:
        return run(["gh", *arguments], check=True, capture_output=True).stdout


def validated_assets(directory: Path, tag: str) -> list[Path]:
    """Verify the exact expected release files and their recorded hashes."""
    if STABLE_TAG.fullmatch(tag) is None:
        raise ValueError("invalid stable release tag")
    stem = f"vectorstore_ai-{tag[1:]}"
    archives = sorted(
        (directory / f"{stem}-py3-none-any.whl", directory / f"{stem}.tar.gz")
    )
    checksums = directory / "SHA256SUMS"
    notes = directory / "RELEASE_NOTES.md"
    expected = {*archives, checksums, notes}
    if set(directory.iterdir()) != expected or any(
        path.is_symlink() or not path.is_file() for path in expected
    ):
        raise ValueError(
            "release bundle must contain exactly its archives, checksums, and notes"
        )
    actual = "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in archives
    )
    if checksums.read_text() != actual:
        raise ValueError("release artifact checksums do not match")
    if not notes.read_text().startswith(f"## [{tag[1:]}] - "):
        raise ValueError("release notes do not match the tag")
    return [*archives, checksums]


def publish_release(
    client: GitHubReleaseClient, directory: Path, tag: str, commit: str
) -> None:
    """Resume a matching draft or recognize an identical completed release.

    All existing assets are checked before any upload. Published releases and
    conflicting drafts are never overwritten, deleted, or silently repaired.
    """
    assets = validated_assets(directory, tag)
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or client.tag_commit(tag) != commit
    ):
        raise ValueError("remote release tag does not point at the validated commit")
    body = (directory / "RELEASE_NOTES.md").read_text().rstrip()
    body += f"\n\n<!-- vectorstore-ai-commit: {commit} -->\n"
    release = client.release(tag)
    if release is None:
        with TemporaryDirectory(prefix="vectorstore-release-notes-") as temporary:
            notes = Path(temporary) / "notes.md"
            notes.write_text(body, encoding="utf-8")
            client.create_draft(tag, commit, notes)
        release = client.release(tag)
    if release is None or str(release.get("body", "")).strip() != body.strip():
        raise ValueError("existing release notes or commit differ from this release")

    def missing_assets(current: dict[str, Any]) -> list[Path]:
        existing = current["assets"]
        by_name = {str(asset["name"]): asset for asset in existing}
        if len(by_name) != len(existing) or set(by_name) - {
            path.name for path in assets
        }:
            raise ValueError("release contains unexpected or duplicate assets")
        missing = []
        for path in assets:
            if path.name not in by_name:
                missing.append(path)
            elif client.asset_bytes(int(by_name[path.name]["id"])) != path.read_bytes():
                raise ValueError(f"existing release asset conflicts: {path.name}")
        return missing

    missing = missing_assets(release)
    if not release["draft"]:
        if missing:
            raise ValueError("published release is missing expected assets")
        print(f"{tag} already published with identical assets")
        return
    for path in missing:
        client.upload(tag, path)
    current = client.release(tag)
    if (
        current is None
        or str(current.get("body", "")).strip() != body.strip()
        or missing_assets(current)
    ):
        raise ValueError(
            "release changed or asset verification failed before publication"
        )
    if client.tag_commit(tag) != commit:
        raise ValueError("release tag moved before publication")
    if current["draft"]:
        client.publish(tag)
    print(f"Published {tag} with verified assets")


def main() -> None:
    """Publish artifacts from the same successful workflow run."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    publish_release(
        GitHubReleaseClient(args.repository), args.directory, args.tag, args.commit
    )


if __name__ == "__main__":
    main()
