# Versioning

FastPKI uses [Semantic Versioning](https://semver.org/) and [bump-my-version](https://github.com/callowayproject/bump-my-version) for automated version management.

## Current Version

To display the current version:

```bash
uv run bump-my-version show current_version
```

The single source of truth is the `version` field under `[project]` in
`pyproject.toml` (PEP 621). bump-my-version reads and updates it directly, so
there is no second copy to keep in sync.

## Bumping the Version

Use the Makefile targets to bump the version:

```bash
# Patch release (v0.1.0 -> v0.1.1)
make bump-patch

# Minor release (v0.1.0 -> v0.2.0)
make bump-minor

# Major release (v0.1.0 -> v1.0.0)
make bump-major
```

Each bump will:

1. Update the `version` field under `[project]` in `pyproject.toml` (no `v` prefix, e.g. `0.2.0`)
2. Create a git commit with the message `release: Bump version 0.1.0 -> 0.2.0`
3. Create a git tag with the `v` prefix (e.g., `v0.2.0`)

A bump requires a clean working tree (`allow_dirty = false`). Preview any bump
without changing anything:

```bash
uv run bump-my-version bump --dry-run --verbose patch
```

!!! note
    Bumps do **not** push to the remote automatically. Run `git push && git push --tags` when ready.

## Configuration

The bump-my-version configuration lives in `pyproject.toml`:

```toml
[tool.bumpversion]
# current_version is omitted on purpose: bump-my-version reads and updates the
# PEP 621 [project].version, so there is a single source of truth.
allow_dirty = false
commit = true
tag = true
tag_name = "v{new_version}"
tag_message = "release: v{new_version}"
message = "release: Bump version {current_version} -> {new_version}"
```

Because `current_version` is omitted, the `[project].version` value is the only
place the version lives — there is no separate config field that can drift out
of sync.

### Adding Version to More Files

To stamp the version into additional files, add a `[[tool.bumpversion.files]]`
table per file:

```toml
[[tool.bumpversion.files]]
filename = "app/__init__.py"
search = '__version__ = "{current_version}"'
replace = '__version__ = "{new_version}"'
```

## Docker Image Versioning

The Dockerfile accepts a `VERSION` build argument that is stored as an OCI label:

```bash
docker build -f docker/Dockerfile --build-arg VERSION=v0.1.0 -t fastpki:v0.1.0 .
```

This sets the `org.opencontainers.image.version` label on the image, which can be inspected with:

```bash
docker inspect fastpki:v0.1.0 --format '{{ index .Config.Labels "org.opencontainers.image.version" }}'
```
