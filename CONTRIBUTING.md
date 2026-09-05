# Contributing

Use Python 3.14 and uv 0.12.10 (the CI version). Commands below run from the
repository root. Ubuntu 24.04 is the validated CI platform; other platforms
are not yet a tested compatibility promise.

## Development and dependencies

```bash
uv python install
uv sync --locked
```

`dev` is the default group and includes `test`, Ruff, ty, mypy, and dotenv for
demos. `test` contains pytest, pytest-asyncio, and pytest-socket. `backend-test`
adds the OpenTelemetry SDK only for tests; `build` contains Hatchling and Twine.
Runtime integrations remain opt-in extras. Avoid `--all-extras` for routine
work: the `local` extra can download a large Torch/CUDA dependency stack.

Update dependency declarations and `uv.lock` together. Use `uv lock` for an
intentional dependency change and `uv sync --locked` otherwise. Dependabot is
configured for weekly uv and GitHub Actions PRs; do not auto-merge major updates
or weaken checks to make an update pass. Action versions are full-SHA pinned.
When updating uv, keep both workflow pins and these instructions synchronized.

## Quality and offline tests

Use the non-model integrations for the same quality environment as CI:

```bash
uv sync --locked --group backend-test --extra chroma --extra faiss --extra otel --extra postgres --extra azure-sql
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync mypy
uv run --no-sync python -c 'from tiktoken import get_encoding; get_encoding("cl100k_base")'
uv run --no-sync pytest -m "not local_model and not postgres_integration" --disable-socket --allow-unix-socket
```

On Ubuntu, the Azure SQL driver needs `libltdl7`, `libkrb5-3`, and
`libgssapi-krb5-2` (`sudo apt-get install` these if absent). Optional integration
tests skip when their dependencies are absent; the backend CI job explicitly
requires imports so a missing integration cannot silently turn that job green.
Ordinary tests never call OpenAI or download models. Tiktoken's first asset
download occurs before sockets are disabled; Unix sockets remain allowed for
local runtime machinery. CI also disables Hugging Face network access and
Chroma telemetry. Dependency and tokenizer setup still need network access.

For the minimal core dependency tier:

```bash
uv sync --locked --no-default-groups --group test
uv run --no-sync python -c 'from tiktoken import get_encoding; get_encoding("cl100k_base")'
uv run --no-sync pytest -m "not local_model and not postgres_integration" --disable-socket --allow-unix-socket
```

Run `actionlint` 1.7.12 when editing workflows. CI verifies the release archive
checksum before running that tool; the pinned URL and checksum are in `ci.yml`.

## PostgreSQL integration tests

Use a disposable PostgreSQL 18 database, never production. Tests create a
random `vectorstore_test_*` schema per test and drop only that schema afterward;
the test identity must be allowed to create schemas. They cover real FTS,
scope/filter pushdown, lifecycle updates, rollback, and concurrent durable
budget reservations across independent sessions.

```bash
uv sync --locked --no-default-groups --group test --extra postgres
export VECTORSTORE_TEST_POSTGRES_DSN='postgresql://test_user:test_password@127.0.0.1:5432/test_database'
VECTORSTORE_REQUIRE_POSTGRES_TESTS=1 uv run --no-sync pytest -m postgres_integration
```

The dedicated DSN is intentionally distinct from the application's
`POSTGRES_CONNECTIONSTRING`. CI provisions its own PostgreSQL service and
requires these tests. Without a DSN they skip during ordinary local runs.
Azure SQL behavior remains covered through injected fake connections, plus
an installed-driver import check; CI does not provision Azure infrastructure.

## Optional real-model tests (CPU)

The manual **Local model (CPU)** GitHub workflow downloads MiniLM during setup,
then blocks network during tests. It is outside the required PR/release gate.
Its fresh consumer resolution intentionally does not use the project's
CUDA-oriented lock resolution. Run the equivalent in a separate environment:

```bash
uv venv --python 3.14 /tmp/vectorstore-model-test
VIRTUAL_ENV=/tmp/vectorstore-model-test uv pip install --torch-backend cpu --group test '.[local]'
/tmp/vectorstore-model-test/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2", device="cpu")'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/vectorstore-model-test/bin/python -m pytest tests/test_sentence_transformers.py -m local_model --disable-socket --allow-unix-socket
```

Use a fresh path if that environment already exists. Do not rewrite `uv.lock`
to select CPU-only wheels for every consumer.

## Packaging and pull requests

See [RELEASING](docs/RELEASING.md) for the locked build and isolated-consumer
checks. Keep changes focused, add regression tests, and update the README,
examples, and `[Unreleased]` notes when behavior changes. Preserve source
adaptation, provider, storage, catalog, retrieval, and observation boundaries;
inject external clients and keep optional imports lazy. Preserve the invariant
that vectors from distinct embedding spaces are never compared.

Do not commit credentials, private corpora, local vector databases, or editor
state. Use synthetic fixtures. Report vulnerabilities privately according to
[SECURITY.md](SECURITY.md), not in a public issue or test fixture.
