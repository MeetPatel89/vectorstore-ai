# Repository guidance

Read the relevant code and tests before changing behavior. Keep changes scoped
and preserve unrelated worktree edits. Do not publish, push, tag, change GitHub
settings, or touch downstream applications without explicit authorization.

## Architecture and design

- Preserve the boundaries between source adapters, immutable record/chunk
  models, chunkers, embedding providers, stores, catalogs, retrieval, and
  observation. One vector store belongs to one embedding space.
- Keep optional integrations lazy and out of core dependency requirements.
  Inject external clients and use deterministic fakes for ordinary tests.
- Prefer small cohesive functions/classes and composition. Introduce protocols
  only for real substitution boundaries; do not add speculative abstractions.
- Keep invariants and mutation encapsulated; do not leak mutable internal state.
  Make error handling and dependency direction explicit.

## Validation and documentation

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for locked dependency groups and exact
quality/test commands. Run Ruff format/check, ty, mypy, and relevant tests.
Changes to CI or release tools also require actionlint and release-tool tests;
package changes require wheel/sdist content and isolated-consumer validation.
Use only disposable databases for integration tests. Do not run a real-model
download merely to validate an unrelated edit.

Perform a README impact check for every change. Update README and related docs
alongside user-visible behavior, dependencies, commands, defaults, limitations,
and safety boundaries; verify examples against code. Avoid claims about
unexecuted checks or unpublished releases. Record user-visible changes under
`[Unreleased]`; version, dated notes, and tags must match when releasing.

Use synthetic fixtures. Never include private corpora, credentials, local editor
state, notebooks, or planning files in distribution artifacts. Keep Cursor's
README-maintenance and proportional Python design guidance consistent with
these instructions; Cursor-specific rule files belong in `.cursor/rules/`.
