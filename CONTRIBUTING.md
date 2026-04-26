# Contributing

Thanks for your interest in Agent Memory! Here's how to contribute.

## Quick Start

```bash
git clone https://github.com/dorukardahan/noldo-memory.git
cd noldo-memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Development Workflow

1. **Fork & branch** from `main`
2. **Write tests** for new features or bug fixes
3. **Run checks locally** before pushing:
   ```bash
   ruff check agent_memory/ tests/ scripts/
   python -m pytest tests/ -v
   ```
4. **Open a PR** with a clear description

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): add new feature`
- `fix(scope): fix a bug`
- `docs: update README`
- `chore: maintenance task`
- `test: add or fix tests`

## Pull Request Guidelines

- One concern per PR (ideally under 200 lines)
- Include tests for new endpoints or logic changes
- Update README if you add/change API endpoints or config options
- All CI checks must pass before merge

## Releases

Releases are manual. Do not add semantic-release or auto-release workflows.

Before tagging a release:

1. Update the version in all runtime manifests:
   - `pyproject.toml`
   - `agent_memory/__init__.py`
   - `plugin/package.json`
   - `hooks/package.json`
   - `plugin/index.js`
2. Update `CHANGELOG.md`.
3. Run CI locally where practical:
   ```bash
   ruff check agent_memory/ tests/ scripts/
   python -m pytest tests/ -v
   ```
4. Create the git tag and GitHub release manually.

## Code Style

- Python 3.10+
- [ruff](https://docs.astral.sh/ruff/) for linting (config in `pyproject.toml`)
- Line length: 120 characters
- Type hints encouraged but not enforced

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_api.py -v

# Quick check (stop on first failure)
python -m pytest tests/ -x -q
```

Tests use an in-memory SQLite database and mock the embedding service. No external services required.

## Reporting Issues

- Search existing issues first
- Include: what happened, what you expected, steps to reproduce
- For bugs: Python version, OS, and relevant config

## Security

If you find a security vulnerability, please do **not** open a public issue. Contact the maintainer directly.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
