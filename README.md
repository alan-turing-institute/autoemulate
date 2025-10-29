# autox

## Workspace subpackage virtual environments
If using a virtual workspace for more than one subpackage, then:
- Run `uv sync --all-packages` at the top-level

## Distinct subpackage virtual environments
If using a distinct venv for a subpackage (e.g. `autoemulate/`), but would like a discoverable venv at the top-level then:
- Create `venv`:
    - `uv venv --python=3.11 --path=.venv_autoemulate`
- Change directory:
    - `cd autox/autoemulate/`
- Activate the venv  (remain in the subpackage directory):
    - `source ../../.venv_autoemulate/bin/activate`
- Sync the venv with the subpackage requirements (remain in the subpackage directory and use active flag to indicate to use the active venv):
    - `uv sync --extra dev --extra spatiotemporal --active`