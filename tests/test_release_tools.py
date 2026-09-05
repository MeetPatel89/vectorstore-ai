from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Any, override

import pytest

from scripts.publish_release import GitHubReleaseClient, publish_release
from scripts.recreate_release_tag import recreate_release_tag
from scripts.release_metadata import check_main, release_notes

COMMIT = "a" * 40
TAG = "v0.2.0"
NOTES = "## [0.2.0] - 2026-01-01\n\n### Added\n\n- Hybrid retrieval.\n"


@pytest.mark.parametrize(
    "tag", ["v0.2", "v00.2.0", "v0.2.0rc1", "v0.2.0;echo bad", "0.2.0"]
)
def test_release_rejects_malformed_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="syntax"):
        release_notes(tag, "0.2.0", NOTES)


def test_release_requires_matching_version_and_dated_notes() -> None:
    with pytest.raises(ValueError, match="version"):
        release_notes(TAG, "0.1.0", NOTES)
    for notes in (
        "## [Unreleased]\n\n- Something\n",
        NOTES + NOTES,
        NOTES.replace("2026-01-01", "2026-02-30"),
        "## [0.2.0] - 2026-01-01\n",
    ):
        with pytest.raises(ValueError):
            release_notes(TAG, "0.2.0", notes)
    assert (
        release_notes(
            TAG,
            "0.2.0",
            "# Changelog\n\n## [Unreleased]\n\n" + NOTES + "\n## [0.1.0]\nOld",
        )
        == NOTES
    )


def test_release_commit_must_be_contained_in_main(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        run(
            [
                "git",
                "-c",
                "user.name=CI Test",
                "-c",
                "user.email=ci@example.invalid",
                *arguments,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-b", "main")
    git("commit", "--allow-empty", "-m", "main")
    assert len(check_main(tmp_path, "main")) == 40
    git("switch", "-c", "feature")
    git("commit", "--allow-empty", "-m", "unmerged")
    with pytest.raises(ValueError, match="contained"):
        check_main(tmp_path, "main")


class FakeGitHub(GitHubReleaseClient):
    def __init__(self) -> None:
        super().__init__("owner/repository")
        self.commit = COMMIT
        self.state: dict[str, Any] | None = None
        self.contents: dict[int, bytes] = {}
        self.uploads: list[str] = []
        self.publications = 0
        self.fail_upload: str | None = None

    @override
    def tag_commit(self, tag: str) -> str:
        return self.commit

    @override
    def release(self, tag: str) -> dict[str, Any] | None:
        return deepcopy(self.state)

    @override
    def create_draft(self, tag: str, commit: str, notes: Path) -> None:
        self.state = {"body": notes.read_text(), "draft": True, "assets": []}

    @override
    def asset_bytes(self, asset_id: int) -> bytes:
        return self.contents[asset_id]

    @override
    def upload(self, tag: str, asset: Path) -> None:
        if self.fail_upload == asset.name:
            raise RuntimeError("interrupted upload")
        assert self.state is not None
        asset_id = len(self.contents) + 1
        self.contents[asset_id] = asset.read_bytes()
        self.state["assets"].append({"name": asset.name, "id": asset_id})
        self.uploads.append(asset.name)

    @override
    def publish(self, tag: str) -> None:
        assert self.state is not None
        self.state["draft"] = False
        self.publications += 1


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    archives = [
        tmp_path / "vectorstore_ai-0.2.0-py3-none-any.whl",
        tmp_path / "vectorstore_ai-0.2.0.tar.gz",
    ]
    for archive in archives:
        archive.write_bytes(archive.name.encode())
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in archives
        )
    )
    (tmp_path / "RELEASE_NOTES.md").write_text(NOTES)
    return tmp_path


def test_publish_and_identical_rerun(bundle: Path) -> None:
    client = FakeGitHub()
    publish_release(client, bundle, TAG, COMMIT)
    publish_release(client, bundle, TAG, COMMIT)
    assert len(client.uploads) == 3
    assert client.publications == 1


def test_interrupted_draft_resumes_without_replacing_assets(bundle: Path) -> None:
    client = FakeGitHub()
    client.fail_upload = "SHA256SUMS"
    with pytest.raises(RuntimeError, match="interrupted"):
        publish_release(client, bundle, TAG, COMMIT)
    assert client.publications == 0
    client.fail_upload = None
    publish_release(client, bundle, TAG, COMMIT)
    assert len(client.uploads) == 3
    assert client.publications == 1


def test_remote_asset_conflict_fails_before_any_new_upload(bundle: Path) -> None:
    client = FakeGitHub()
    client.fail_upload = "SHA256SUMS"
    with pytest.raises(RuntimeError):
        publish_release(client, bundle, TAG, COMMIT)
    client.contents[1] = b"different content"
    client.fail_upload = None
    with pytest.raises(ValueError, match="conflicts"):
        publish_release(client, bundle, TAG, COMMIT)
    assert len(client.uploads) == 2
    assert client.publications == 0


def test_moved_tag_and_corrupt_bundle_cannot_create_a_release(bundle: Path) -> None:
    client = FakeGitHub()
    client.commit = "b" * 40
    with pytest.raises(ValueError, match="commit"):
        publish_release(client, bundle, TAG, COMMIT)
    assert client.state is None
    client.commit = COMMIT
    (bundle / "vectorstore_ai-0.2.0.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksums"):
        publish_release(client, bundle, TAG, COMMIT)
    assert client.state is None


def test_incomplete_published_release_is_not_repaired(bundle: Path) -> None:
    client = FakeGitHub()
    publish_release(client, bundle, TAG, COMMIT)
    assert client.state is not None
    client.state["assets"].pop()
    with pytest.raises(ValueError, match="missing"):
        publish_release(client, bundle, TAG, COMMIT)
    assert len(client.uploads) == 3


@pytest.mark.parametrize("draft_exists", [True, False])
def test_client_looks_up_drafts_when_rest_tag_is_missing(
    monkeypatch: pytest.MonkeyPatch, draft_exists: bool
) -> None:
    calls: list[tuple[str, ...]] = []

    def command(*arguments: str) -> bytes:
        calls.append(arguments)
        if arguments[1].endswith(f"/tags/{TAG}"):
            raise CalledProcessError(1, arguments, stderr=b"gh: Not Found (HTTP 404)")
        if arguments[1] == "graphql":
            assert "--jq" not in arguments
            release = {"databaseId": 42} if draft_exists else None
            return json.dumps({"data": {"repository": {"release": release}}}).encode()
        assert arguments == ("api", "repos/owner/repository/releases/42")
        return b'{"draft": true, "body": "notes", "assets": []}'

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    result = GitHubReleaseClient("owner/repository").release(TAG)
    assert (result is not None) == draft_exists
    assert len(calls) == (3 if draft_exists else 2)


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"\n",
        b"not JSON",
        b"\xff",
        b"null",
        b"[]",
        b"{}",
        b'{"data": null}',
        b'{"data": {"repository": null}}',
        b'{"data": {"repository": {}}}',
        b'{"errors": [{"message": "Forbidden"}], '
        b'"data": {"repository": {"release": null}}}',
    ],
)
def test_client_rejects_invalid_release_lookup_responses(
    monkeypatch: pytest.MonkeyPatch, response: bytes
) -> None:
    calls: list[tuple[str, ...]] = []

    def command(*arguments: str) -> bytes:
        calls.append(arguments)
        if len(calls) == 1:
            raise CalledProcessError(1, arguments, stderr=b"gh: Not Found (HTTP 404)")
        assert arguments[1] == "graphql"
        return response

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    with pytest.raises(ValueError, match="release lookup"):
        GitHubReleaseClient("owner/repository").release(TAG)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "release",
    [
        {},
        [],
        "42",
        *({"databaseId": value} for value in (None, True, False, 0, -1, 42.5, "42")),
    ],
)
def test_client_rejects_invalid_draft_release_ids(
    monkeypatch: pytest.MonkeyPatch, release: object
) -> None:
    calls: list[tuple[str, ...]] = []

    def command(*arguments: str) -> bytes:
        calls.append(arguments)
        if len(calls) == 1:
            raise CalledProcessError(1, arguments, stderr=b"gh: Not Found (HTTP 404)")
        assert arguments[1] == "graphql"
        return json.dumps({"data": {"repository": {"release": release}}}).encode()

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    with pytest.raises(ValueError, match="invalid draft release ID"):
        GitHubReleaseClient("owner/repository").release(TAG)
    assert len(calls) == 2


def test_client_propagates_graphql_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = CalledProcessError(1, ["gh", "api", "graphql"], stderr=b"Forbidden")

    def command(*arguments: str) -> bytes:
        if arguments[1] == "graphql":
            raise failure
        raise CalledProcessError(1, arguments, stderr=b"gh: Not Found (HTTP 404)")

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    with pytest.raises(CalledProcessError) as caught:
        GitHubReleaseClient("owner/repository").release(TAG)
    assert caught.value is failure


def test_client_does_not_treat_api_failure_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def command(*arguments: str) -> bytes:
        raise CalledProcessError(1, arguments, stderr=b"gh: Forbidden (HTTP 403)")

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    with pytest.raises(CalledProcessError):
        GitHubReleaseClient("owner/repository").release(TAG)


def test_client_resolves_the_tag_namespace_not_a_same_named_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def command(*arguments: str) -> bytes:
        assert arguments == (
            "api",
            f"repos/owner/repository/commits/refs%2Ftags%2F{TAG}",
            "--jq",
            ".sha",
        )
        return f"{COMMIT}\n".encode()

    monkeypatch.setattr(GitHubReleaseClient, "_run", staticmethod(command))
    assert GitHubReleaseClient("owner/repository").tag_commit(TAG) == COMMIT


def test_conflicting_release_body_is_not_overwritten(bundle: Path) -> None:
    client = FakeGitHub()
    client.state = {"draft": True, "body": "unrelated release", "assets": []}
    with pytest.raises(ValueError, match="notes or commit"):
        publish_release(client, bundle, TAG, COMMIT)
    assert not client.uploads
    assert client.publications == 0


def _git(cwd: Path, *arguments: str) -> str:
    return run(
        [
            "git",
            "-c",
            "user.name=CI Test",
            "-c",
            "user.email=ci@example.invalid",
            *arguments,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _release_clone(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    run(
        ["git", "init", "--bare", "-b", "main", str(origin)],
        check=True,
        capture_output=True,
    )
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "CI Test")
    _git(repo, "config", "user.email", "ci@example.invalid")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "vectorstore-ai"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n" + NOTES,
        encoding="utf-8",
    )
    _git(repo, "add", "pyproject.toml", "CHANGELOG.md")
    _git(repo, "commit", "-m", "release notes")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-u", "origin", "main")
    return repo


def _tagged_commit(repository: Path, tag: str) -> str:
    return _git(repository, "rev-parse", f"{tag}^{{commit}}")


def _origin(repository: Path) -> Path:
    return repository.parent / "origin.git"


def test_new_release_tag_is_pushed_to_origin(tmp_path: Path) -> None:
    repo = _release_clone(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    recreate_release_tag(repo, TAG)
    assert _tagged_commit(repo, TAG) == head
    assert _tagged_commit(_origin(repo), TAG) == head


def test_existing_tag_is_not_replaced_without_an_explicit_flag(
    tmp_path: Path,
) -> None:
    repo = _release_clone(tmp_path)
    recreate_release_tag(repo, TAG)
    first = _tagged_commit(_origin(repo), TAG)
    (repo / "note").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "note")
    _git(repo, "commit", "-m", "later")
    _git(repo, "push", "origin", "main")
    with pytest.raises(ValueError, match="already exists"):
        recreate_release_tag(repo, TAG)
    assert _tagged_commit(_origin(repo), TAG) == first


def test_replace_existing_moves_the_tag_after_validation(tmp_path: Path) -> None:
    repo = _release_clone(tmp_path)
    recreate_release_tag(repo, TAG)
    (repo / "note").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "note")
    _git(repo, "commit", "-m", "later")
    _git(repo, "push", "origin", "main")
    head = _git(repo, "rev-parse", "HEAD")
    recreate_release_tag(repo, TAG, replace_existing=True)
    assert _tagged_commit(repo, TAG) == head
    assert _tagged_commit(_origin(repo), TAG) == head


def test_failed_validation_does_not_delete_an_existing_tag(tmp_path: Path) -> None:
    repo = _release_clone(tmp_path)
    recreate_release_tag(repo, TAG)
    first = _tagged_commit(_origin(repo), TAG)
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n", encoding="utf-8"
    )
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-m", "drop notes")
    _git(repo, "push", "origin", "main")
    with pytest.raises(ValueError, match="dated release entry"):
        recreate_release_tag(repo, TAG, replace_existing=True)
    assert _tagged_commit(_origin(repo), TAG) == first


def test_dirty_tree_and_unpushed_head_are_rejected(tmp_path: Path) -> None:
    repo = _release_clone(tmp_path)
    (repo / "dirty").write_text("nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        recreate_release_tag(repo, TAG)
    (repo / "dirty").unlink()
    (repo / "local").write_text("ahead\n", encoding="utf-8")
    _git(repo, "add", "local")
    _git(repo, "commit", "-m", "unpushed")
    with pytest.raises(ValueError, match="HEAD must match"):
        recreate_release_tag(repo, TAG)
