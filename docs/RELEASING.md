# Release management

Distribution is through GitHub Releases, not PyPI. The package remains `0.1.0`
until a separate release PR prepares `0.2.0`. Adding this infrastructure does
not create a tag, publish a release, or migrate a downstream application.

## CI and trust boundaries

PRs to `main`, pushes to `main` and `v*` tags, merge queues, and manual dispatches
run the same required validation on Ubuntu 24.04 / Python 3.14:

| Job | Evidence |
| --- | --- |
| Quality | Locked Ruff/ty/mypy; checksum-pinned actionlint; tag metadata/ancestry |
| Tests (core) | Minimal runtime + test group; Python network blocked |
| Tests (backends) | Chroma, FAISS, OTel SDK, Psycopg, Azure driver; network blocked |
| PostgreSQL | Disposable PostgreSQL 18; real catalog and concurrent budget tests |
| Package | Locked build tools, strict metadata/content checks, wheel + source consumer tests and source-archive demos |
| CI | Fails unless every required validation job succeeds |

The wheel is installed independently with each non-model extra; core wheel and
source consumers exercise NumPy persistence, SQLite ingestion/lifecycle, FTS,
hybrid retrieval, and scope enforcement outside the checkout. Chroma and FAISS
also perform real local storage/search operations. SQL extras smoke-test driver
imports; real PostgreSQL behavior is tested in its separate job. Azure SQL has
no live-service gate. The optional CPU MiniLM workflow uses fresh consumer
resolution and does not block releases.

All third-party actions are SHA-pinned. Checks have read-only permissions,
timeouts, and locked project environments; only PR superseded runs are
cancelled. The release job receives `contents: write` only after the aggregate
gate succeeds on a tag push in the upstream repository. It downloads the
validated artifacts from the same run and checks out the validated commit
only to execute the standard-library release helper. It never installs/builds
the package or executes its runtime under write permission. No PyPI token or
other publishing secret is required; `GITHUB_TOKEN` is scoped to that step.

## One-time repository-owner setup

These are GitHub settings, not settings that workflow files can activate:

1. Enable Actions and allow the pinned checkout/setup-uv/upload/download actions.
   Keep the repository's default workflow token read-only; allow the release
   job's explicit write permission. Fork PRs must not receive write tokens or
   repository secrets.
2. Protect `main`: require a PR, review, and the aggregate **CI** status from
   GitHub Actions; block force pushes/deletion. Run CI once so the exact check
   can be selected. Include `merge_group` support if enabling a merge queue.
3. Create a ruleset for `v*` tags: restrict creation to release maintainers and
   prohibit updates/deletion. Keep any bypass actors narrowly scoped. The
   workflow rejects tags whose commit is not contained in `origin/main`, but
   tag protection is still essential because workflows are code in the tag.
4. Enable dependency graph, Dependabot alerts/security updates, and private
   vulnerability reporting. Weekly uv and Actions update PRs are already
   configured in `.github/dependabot.yml`; review rather than auto-merge them.
5. Optionally enable immutable releases after reviewing GitHub's setting.
   The publisher uploads into a draft before publication and never overwrites
   published assets. No CODEOWNERS file is added until ownership is agreed.

Repository settings and a hosted Actions run still need owner verification.
Local checks alone do not prove those settings are active.

## Build locally

Use a clean checkout and the CI uv version, 0.12.10:

```bash
uv sync --locked --only-group build
uv build --no-sources --no-build-isolation
uv run --no-sync twine check --strict dist/*
uv run --no-sync python -m scripts.check_dist dist
```

The build group locks the actual Hatchling/Twine versions; the build-system
range also supports source consumers without a uv lock. `--no-sources` prevents
workspace source overrides. Both archives explicitly use core metadata 2.4
(including SPDX license fields), compatible with the locked Twine validator.
The archive check requires correct version/license
metadata, `py.typed`, and packaged examples, rejects local/editor/planning data,
and writes `dist/SHA256SUMS`. Use a clean `dist/` directory; stale archives fail
validation rather than being silently published.

For an installed core wheel check, use an absolute wheel path and run outside
the checkout (substitute the checkout's absolute path for the smoke script):

```bash
cd /tmp
uv run --isolated --no-project --python 3.14 --with /absolute/path/to/dist/vectorstore_ai-0.1.0-py3-none-any.whl python /absolute/path/to/vectorstore-ai/tests/package_smoke.py 0.1.0
```

The Package job contains the full wheel/extras/source smoke and demo loop.
`uv sync --locked` restores the normal development environment after building.

## Prepare and publish a release

1. Open a release PR: bump `project.version`, run `uv lock`, and move the
   appropriate `[Unreleased]` notes into a nonempty dated version section.
   Use stable `MAJOR.MINOR.PATCH` versions only; this workflow deliberately
   rejects prerelease tags. Call out breaking changes even before 1.0.
2. Merge after **CI** succeeds. Fetch `origin/main`, check out the intended
   merged commit, and validate locally (the first intended version is shown):

   ```bash
   uv run --isolated --no-project --python 3.14 python -m scripts.release_metadata v0.2.0
   ```

3. An authorized maintainer creates an annotated `v0.2.0` tag at that commit and
   pushes that tag. Do not tag a feature branch, move an existing tag, or use
   `workflow_dispatch` as a substitute for the tag push.
4. Tag CI repeats all checks and requires exact agreement between tag, package
   version, dated changelog notes, and `main` ancestry. Only then does the
   release job publish the wheel, source archive, and `SHA256SUMS`, with the
   extracted changelog notes as the release body. A commit marker ties the
   body to the validated source revision.
5. Inspect the published assets/checksums and perform a consumer install.
   Update downstream applications in their own reviewed change. For
   fieldguide-ai, the later migration must enforce `RetrievalScope`, replace
   duplicated retrieval/pipeline code, and add consumer regression tests.

CI distribution artifacts expire after 30 days; test reports after 7 days.
GitHub Release assets are the durable distribution channel.

## Retries and conflicts

The publisher verifies the remote tag commit, local checksums, release body,
and bytes of existing assets before mutation. Matching partial drafts resume
by uploading only missing assets. A matching complete published release is a
no-op. Unexpected, duplicate, conflicting, or missing published assets fail;
nothing is clobbered or deleted. The tag is checked again before publication.

Retry the failed release job within the same Actions run so it reuses the
exact artifacts. Rebuilding may produce different bytes even at the same
version (for example a fresh source-consumer dependency resolution); the
publisher will refuse to replace already uploaded conflicting bytes. If a
draft conflicts, investigate it manually. Do not delete assets or move tags
as routine recovery; publish a new version when correcting a public release.

## Consumer installation

After a release exists, download its wheel and `SHA256SUMS` from the release
page. To verify the wheel alone when the source archive is not downloaded,
use `sha256sum --ignore-missing --check SHA256SUMS`, then install it with
`uv pip install ./vectorstore_ai-VERSION-py3-none-any.whl` (or `pip install`).
For optional integrations use a direct requirement such as
`vectorstore-ai[postgres] @ file:///absolute/path/to/the.whl`. Pin versions and
review checksum changes; never rely on a mutable branch for production.

Git consumers can instead declare the package in `project.dependencies` and
use `[tool.uv.sources]` with the repository URL and a reviewed `tag` or full
`rev`, then commit their own `uv.lock`. A Git installation builds from source;
it is not the same byte-level artifact as the validated release wheel.
