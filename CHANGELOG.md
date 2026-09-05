# Changelog

User-visible changes are recorded here. The package is pre-1.0; minor releases
may include breaking changes, which must be called out explicitly. Release
headings use `## [MAJOR.MINOR.PATCH] - YYYY-MM-DD` and contain dated, nonempty
notes before a matching `vMAJOR.MINOR.PATCH` tag can publish.

## [Unreleased]

### Added

- Structured, dense, lexical, and hybrid retrieval; SQLite, PostgreSQL, and
  Azure SQL catalogs; provider routing and durable budgets; lifecycle-aware
  Markdown/CSV/JSON ingestion; optional OpenTelemetry observation (Phases 1–6).
- Tiered GitHub Actions checks, real PostgreSQL integration coverage, and
  manually dispatched CPU MiniLM tests.
- Validated wheel/source distributions and resumable, conflict-detecting GitHub
  Releases for stable tags on `main`.
- MIT license, package metadata, `py.typed`, contributor/security/release guides,
  and weekly dependency-update PR configuration.
- Synthetic sample corpus and `--corpus` overrides for all retrieval demos.

### Changed

- `python-dotenv` is development-only; core installs no longer require it.
- Test, backend-test, and build tools have separate locked dependency groups.
- Source archives explicitly exclude local data, notebooks, editor settings,
  credentials, and planning documents.

No release date or `0.2.0` release entry is claimed yet. Prepare that entry and
the version bump together in a later release PR; see [RELEASING](docs/RELEASING.md).
