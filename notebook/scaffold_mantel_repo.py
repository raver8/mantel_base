#!/usr/bin/env python3
"""
Cross-platform scaffold for the MACE-MP-0 multi-layer thin-film project.
Can be pasted into a Jupyter notebook cell or saved as scaffold_mantel_repo.py
and run with `%run scaffold_mantel_repo.py` or `!python scaffold_mantel_repo.py`.
"""
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# User-configurable options
# ---------------------------------------------------------------------------
REPO_NAME = "mantel-thin-film"  # change this if you want a different folder name
PRIVATE = False                 # True -> private GitHub repo, False -> public
CREATE_REMOTE = False           # True -> also run `gh repo create` (requires gh CLI)
# ---------------------------------------------------------------------------


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main():
    root = Path.cwd() / REPO_NAME
    root.mkdir(parents=True, exist_ok=True)

    # --- Directory layout --------------------------------------------------
    for dirname in ["notebooks", "src", "inputs", "outputs", "examples", "tests", "docs"]:
        (root / dirname).mkdir(parents=True, exist_ok=True)

    # --- .gitignore ---------------------------------------------------------
    write_text(
        root / ".gitignore",
        """\
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
.env
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Generated simulation outputs (keep directories with .gitkeep)
outputs/*
!outputs/.gitkeep
examples/*.extxyz
examples/*.lmp
examples/*.traj
*.traj
*.log

# IDE / OS
.vscode/
.idea/
.DS_Store

# Build / packaging
dist/
build/
""",
    )

    # --- Keep empty directories under git ----------------------------------
    for dirname in ["outputs", "examples", "tests", "docs"]:
        (root / dirname / ".gitkeep").touch()

    # --- Python package stubs ----------------------------------------------
    write_text(
        root / "src" / "__init__.py",
        '''"""Source modules for the MACE-MP-0 multi-layer thin-film builder."""
__version__ = "0.1.0"
''',
    )

    write_text(
        root / "src" / "layer_builders.py",
        '''"""HCP/FCC/alloy slab builders used by the thin-film notebook.

Refactor the notebook functions here, e.g., `build_ti_layer`, `build_hf_layer`,
`build_steel_316_layer`, `build_cu_layer`, `build_boron_cap`, `stack_layers`.
"""
from ase import Atoms


def build_ti_layer(size=(3, 3, 2), a=2.95, c=4.68, vacuum=10.0):
    raise NotImplementedError("Migrate build_ti_layer from the notebook.")


def build_hf_layer(size=(3, 3, 2), a=3.19, c=5.05, vacuum=10.0):
    raise NotImplementedError("Migrate build_hf_layer from the notebook.")


def build_steel_316_layer(size=(3, 3, 3), a=3.59, vacuum=10.0):
    raise NotImplementedError("Migrate build_steel_316_layer from the notebook.")


def build_cu_layer(size=(3, 3, 2), a=3.61, vacuum=10.0):
    raise NotImplementedError("Migrate build_cu_layer from the notebook.")


def build_boron_cap(n_boron=10, cell_xy=(10.0, 10.0), z_pos=0.0):
    raise NotImplementedError("Migrate build_boron_cap from the notebook.")


def stack_layers(layers, gap=2.0):
    raise NotImplementedError("Migrate stack_layers from the notebook.")
''',
    )

    write_text(
        root / "src" / "nim_calculator.py",
        '''"""ASE calculator wrapper for remote NVIDIA NIM MACE endpoints.

Migrate the `MACE_NIM_Calculator` class from the notebook here.
"""
from ase.calculators.calculator import Calculator


class MACE_NIM_Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, endpoint_url: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError("Migrate MACE_NIM_Calculator from the notebook.")
''',
    )

    # --- Dependency specs ----------------------------------------------------
    write_text(
        root / "environment.yml",
        """\
name: mantel-thin-film
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pip
  - pytorch
  - ase
  - matplotlib
  - numpy
  - scipy
  - pip:
      - mace-torch
      - requests
""",
    )

    write_text(
        root / "requirements.txt",
        """\
ase>=3.22
mace-torch
requests
numpy
matplotlib
""",
    )

    # --- Inputs --------------------------------------------------------------
    write_text(
        root / "inputs" / "README.md",
        """\
# Inputs

Place reference structures, configuration JSON/YAML files, or SQS alloy cells here.
Generated outputs should go to `outputs/`, not this directory.
""",
    )

    # --- Tests placeholder ---------------------------------------------------
    write_text(
        root / "tests" / "test_placeholder.py",
        '''"""Stub for pytest tests.

Add unit tests for layer builders, stacking, and NIM calculator serialization.
"""


def test_import():
    import src
    assert src.__version__ == "0.1.0"
''',
    )

    # --- README --------------------------------------------------------------
    existing_readme = Path("README.md")
    if existing_readme.exists() and existing_readme.is_file():
        shutil.copy(existing_readme, root / "README.md")
        print(f"Copied existing README.md into {root}")
    elif not (root / "README.md").exists():
        write_text(
            root / "README.md",
            f"""# {REPO_NAME.replace("-", " ").title()}

See the full project README for details.
""",
        )

    # --- Move the main notebook if it is in the current directory ------------
    notebook = Path("mantel_multilayer.ipynb")
    if notebook.exists() and notebook.is_file():
        target = root / "notebooks" / notebook.name
        shutil.move(notebook, target)
        print(f"Moved {notebook.name} -> {target}")

    # --- Git init ------------------------------------------------------------
    try:
        subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial scaffold for MACE-MP-0 thin-film project"],
            cwd=root, check=True,
        )
        print(f"Initialized git repo in {root}")
    except FileNotFoundError:
        print("Warning: 'git' was not found. Skipping git initialization.")
    except subprocess.CalledProcessError as exc:
        print(f"Warning: git command failed: {exc}")

    # --- Optional GitHub remote creation -------------------------------------
    if CREATE_REMOTE:
        try:
            cmd = ["gh", "repo", "create", REPO_NAME, "--source=.", "--push"]
            cmd.append("--private" if PRIVATE else "--public")
            subprocess.run(cmd, cwd=root, check=True)
        except FileNotFoundError:
            print("Warning: 'gh' (GitHub CLI) not found; cannot create remote repo.")
        except subprocess.CalledProcessError as exc:
            print(f"Warning: 'gh repo create' failed: {exc}")

    print("\nScaffold complete.")
    for p in sorted(root.iterdir()):
        print(f"  {p.relative_to(root)}{'/' if p.is_dir() else ''}")


if __name__ == "__main__":
    main()
