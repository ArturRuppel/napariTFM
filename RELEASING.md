# Releasing napariTFM to PyPI

napariTFM publishes via **PyPI Trusted Publishing** (OIDC). No API tokens are
stored anywhere — PyPI trusts releases that come from this repo's GitHub
Actions workflow (`.github/workflows/publish.yml`).

## One-time setup (do this once)

1. Create a PyPI account at https://pypi.org if you don't have one.
2. Go to https://pypi.org/manage/account/publishing/ and add a
   **pending publisher** with:
   - PyPI Project Name: `napariTFM`
   - Owner: `ArturRuppel`
   - Repository name: `napariTFM`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. (Recommended) In the GitHub repo settings → Environments, create an
   environment named `pypi`. You can add reviewers/branch protection here.

## Cutting a release

1. Bump `version` in `pyproject.toml` (PEP 440, e.g. `1.0.1`, `1.1.0`).
2. Commit and tag:
   ```bash
   git commit -am "release: v1.0.1"
   git tag v1.0.1
   git push && git push --tags
   ```
3. On GitHub, draft a **new Release** for that tag and click **Publish**.
   The `Publish to PyPI` workflow builds and uploads automatically.
4. Verify at https://pypi.org/project/napariTFM/ and test:
   ```bash
   pip install napariTFM
   ```

## Build / check locally before releasing

```bash
python -m build              # produces dist/*.whl and dist/*.tar.gz
python -m twine check dist/* # validates metadata
```

## Optional: test on TestPyPI first

Add a separate pending publisher on https://test.pypi.org with the same
settings, temporarily point the workflow at TestPyPI, then:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ napariTFM
```
