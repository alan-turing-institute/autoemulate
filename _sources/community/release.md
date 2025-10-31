# Releasing AutoEmulate

This guide explains how to create new releases of AutoEmulate for maintainers.

## Release Process

AutoEmulate uses GitHub Actions to automatically publish releases to PyPI when a new version tag is pushed. The process is as follows:

1. **Update Version Number**
First, give the package a new version. We recommend [this guide](https://py-pkgs.org/07-releasing-versioning.html) to decide on a version number. Note: you'll need to open a PR to do this.

Update the version in `pyproject.toml`:

```toml
[project]
name = "autoemulate"
version = "X.Y.Z"  # Update this line
```

2. **Create and Push Tag**
Create a new git tag following the same semantic versioning (vX.Y.Z):

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

3. **Create a Release on GitHub**
Go to the [Releases page](https://github.com/alan-turing-institute/autoemulate/releases) of the repository and click "Draft a new release".
Fill in the release title and description, then select the tag you just created (vX.Y.Z). You can also link to the relevant issues or pull requests that are included in this release. There is an option to "Generate release notes" which can help summarize changes since the last release.

4. **Automated Release Process using GitHub Actions**

   When you push a tag matching the pattern 'vX.Y.Z', the release workflow `release.yaml` will automatically:
   - Check out the code
   - Set up Python
   - Install build tools (build, twine)
   - Build the package
   - Publish to PyPI

   The workflow requires a PyPI token stored in the repository secrets as `PYPI_TOKEN`.

## Prerequisites

Before creating a release, ensure:

1. All tests are passing on the main branch
2. Documentation is up to date
3. CHANGELOG.md is updated
4. You have appropriate permissions to push tags to the repository

## Troubleshooting

If the release fails:

1. Check the GitHub Actions logs for errors
2. Verify the PyPI token is correctly set in repository secrets
3. Ensure the version number in `pyproject.toml` matches the git tag
4. Make sure you haven't already published this version to PyPI

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Commit changes
- [ ] Create and push git tag
- [ ] Monitor GitHub Actions workflow
- [ ] Verify package is available on PyPI
- [ ] Test installation from PyPI

## Notes

- The release workflow only triggers on tags matching 'vX.Y.Z'
- Only maintainers with appropriate permissions can create releases
- Each version can only be published to PyPI once
- The workflow uses Python 3.10 for building and publishing