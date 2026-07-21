# Multi-Layer Thin-Film Generator for OVITO

**MACE-MP-0 local or NVIDIA NIM · ASE · HCP/FCC alloy slabs · LAMMPS/extxyz export**

This notebook builds and relaxes a heterogeneous multi-layer thin film (`Ti / Hf / 316 steel / Cu / B cap`) using the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) and the [MACE-MP-0 machine-learned force field](https://github.com/ACEsuit/mace). It can run locally on a CUDA GPU or call a remote NVIDIA NIM endpoint, and it writes `extxyz` + LAMMPS-data files ready for visualization in [OVITO](https://www.ovito.org/) or production simulation in LAMMPS/GROMACS/AMBER/OpenMM-classical workflows.

---

## What problem this solves

Designing coating stacks, radiation-tolerant surfaces, or thermally stable metal/ceramic heterostructures usually requires:

- Layered crystal slabs with realistic HCP/FCC orientations.
- Random alloy substitution (e.g., 316 stainless steel).
- Force-field-based relaxation with fixed substrate regions.
- File formats both human-readable and engine-compatible.

This notebook packages that whole pipeline into a reproducible Colab/local Python flow, with explicit support for both:

1. **Local GPU execution** via `mace-torch` (CUDA, `float32`).
2. **Remote GPU execution** via NVIDIA NIM/BioNeMo-style REST endpoints (HTTP POST, exponential-backoff retries).

The final structure embeds per-element RGB colors for OVITO and exports `mantel_multilayer.extxyz` / `mantel_multilayer.lmp`.

---

## Table of contents

1. [What problem this solves](#what-problem-this-solves)
2. [Technical highlights & MD-engine vocabulary](#technical-highlights--md-engine-vocabulary)
3. [Calculator modes: local MACE vs. NVIDIA NIM](#calculator-modes-local-mace-vs-nvidia-nim)
4. [Quick start](#quick-start)
5. [Outputs](#outputs)
6. [Visualization in OVITO](#visualization-in-ovito)
7. [Limitations & validation notes](#limitations--validation-notes)
8. [Roadmap](#roadmap)
9. [Author context / trajectory](#author-context--trajectory)
10. [References](#references)

---

## Technical highlights & MD-engine vocabulary

This workflow is intentionally constructed around the same core primitives a production MD engine implements. Even though the orchestration is in Python, the data model (cell, PBC, neighbor lists, force-field evaluation, constraints) maps directly onto low-level C++/CUDA engines.

| MD concept | How it appears in this notebook |
|---|---|
| **Force field** | Energy/force evaluation is performed by **MACE-MP-0**, a **machine-learned force field (MLFF)**. It replaces traditional analytic force fields (ReaxFF, EAM, MEAM, Lennard-Jones, etc.) with a many-body equivariant message-passing neural network. |
| **Neighbor lists** | MACE-MP-0 requires environment descriptors over radial/angular neighbors. `mace-torch` builds neighbor lists internally; the NVIDIA NIM backend does the same on the GPU. The ASE calculator interface exposes the resulting energy/forces without hand-rolling Verlet/cell lists in this script. |
| **Periodic boundary conditions (PBC)** | Every slab is generated with a periodic cell, centered along `z`, and cell dimensions are homogenized in `xy` before stacking. The final `extxyz` preserves PBC flags and cell vectors. |
| **Electrostatics / long-range** | MACE-MP-0 folds electrostatic/many-body physics into the learned representation. For classical downstream engines, the exported LAMMPS data file can be rerun with Ewald summation, Particle–Mesh Ewald (PME), or PPPM (for metals/alloys, after assigning appropriate classical potentials). |
| **CUDA / GPU acceleration** | Local mode selects `device="cuda"` when `torch.cuda.is_available()`. Remote NIM mode pushes inference onto NVIDIA-hosted GPU nodes. The script minimizes host↔device transfers by doing all relaxation logic in Python and keeping the heavy vector operations on the GPU. |
| **Constraints / fixed substrate atoms** | The bottom-most layers (Boron cap + Copper) are frozen with `ase.constraints.FixAtoms`, a common boundary condition in thin-film/substrate relaxation runs. |
| **Alloy composition modeling** | The 316 stainless steel layer uses random substitution over an FCC slab to mimic the austenitic composition: Fe/Cr/Ni/Mo/Mn/C. |
| **C++/engine interoperability** | Outputs follow the extxyz and `lammps-data` formats, so the structure can be loaded directly into LAMMPS, GROMACS, AMBER, OpenMM, or OVITO without reformatting. |

---

## Calculator modes: local MACE vs. NVIDIA NIM

### Mode A — Local MACE-MP-0 (recommended for relaxation)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
calc = mace_mp(model="medium", device=device, default_dtype="float32")
```

- Fastest for many FIRE steps.
- Uses the PyTorch/CUDA backend.
- Set `USE_NIM = False`.

### Mode B — NVIDIA NIM remote endpoint

```python
USE_NIM = True
calc = MACE_NIM_Calculator(
    endpoint_url="https://health.api.nvidia.com/v1/biology/mace/mace-mp-0/calculate",
    api_key=userdata.get("NVIDIA_NIM_API_KEY"),
    max_retries=3,
    backoff_factor=1.0,
)
```

- Calls a BioNeMo/NVIDIA NIM MACE microservice over HTTPS.
- Includes exponential-backoff retry logic for transient network failures.
- Relaxation over many NIM calls can be slow/expensive; use for single-point or small validation runs unless you have dedicated throughput.

---

## Quick start

1. Open `mantel_multilayer.ipynb` in Google Colab or a local Jupyter server with a CUDA GPU.
2. Run **Section 1** to install `mace-torch`, `ase`, and `requests`.
3. Choose your calculator in **Section 6**:
   - `USE_NIM = False` (local).
   - `USE_NIM = True` and add `NVIDIA_NIM_API_KEY` to your Colab secrets.
4. Run all cells to build, relax, and export the thin film.
5. Download `mantel_multilayer.extxyz` and `mantel_multilayer.lmp`.

---

## Outputs

| File | Description |
|---|---|
| `mantel_multilayer.extxyz` | ASE-readable extended XYZ with embedded RGB atom colors for OVITO. |
| `mantel_multilayer.lmp` | LAMMPS data file for downstream classical or MLIP simulations. |
| `mantel_relaxation.traj` | ASE trajectory from the FIRE optimizer. |
| `mantel_opt.log` | Optimization log. |

---

## Repository structure

Below is the suggested layout if you promote this notebook to a standalone GitHub repository. The generated simulation files live under `outputs/` and are excluded from Git because of their size.

```text
mantel-thin-film/
├── README.md                       # This file
├── environment.yml                 # Conda / pip dependency spec (mace-torch, ase, requests, matplotlib)
├── .gitignore                      # Ignore outputs/ and __pycache__
├── notebooks/
│   └── mantel_multilayer.ipynb     # Main Colab / Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── layer_builders.py           # Ti, Hf, 316 steel, Cu, B builders (refactored from the notebook)
│   └── nim_calculator.py           # MACE_NIM_Calculator with exponential backoff
├── inputs/
│   └── README.md                   # Placeholder for future DFT reference structures / SQS alloys
└── outputs/                        # Generated artifacts (add to .gitignore)
    ├── mantel_multilayer.extxyz
    ├── mantel_multilayer.lmp
    ├── mantel_relaxation.traj
    └── mantel_opt.log
```

### Notes on the generated artifacts

- **Do not commit `outputs/` directly.** `*.extxyz`, `*.lmp`, and especially `*.traj` can be tens to hundreds of MB.
- For a portfolio repo, either:
  - Provide a small sample output under `examples/`.
  - Attach the full outputs to a GitHub Release.
  - Upload them to Zenodo/Figshare and link from the README.
- The `src/` split is optional for Colab but strongly preferred for a GitHub project; it turns the notebook into importable, testable module code and signals software-engineering maturity.

---


## Visualization in OVITO

1. Load `mantel_multilayer.extxyz` in OVITO.
2. The `Color` array is already attached per atom.
3. Use **Add modification → Color coding → Color (per-atom property)** to visualize the stack:
   - B cap → Blue
   - Ti → White
   - Hf → Green
   - 316 steel (Fe/Cr/Ni/Mo/Mn/C) → gray/green/yellow/purple/orange/black
   - Cu → Brown

---

## Limitations & validation notes

- **316 steel is an approximate austenitic model**, not a DFT-validated or experimental thermodynamic structure. Compositions are assigned by random occupancy on an FCC lattice.
- **Boron cap** is a sparse random 2D layer; it is not a crystalline boride or fully reacted interface.
- **Local relaxation** with MACE-MP-0 medium is sufficient for geometry pre-relaxation but should be validated against DFT or experiments for interface energies and diffusion barriers.
- **Long-range electrostatics**: MACE-MP-0 captures many-body effects, including partial electrostatic character, but it is not a charge equilibration or PME solver. If you move the `.lmp` file into LAMMPS, you will need to select classical potentials and an Ewald/PPPM scheme appropriate to each material.
- **GPU memory**: Local CUDA mode requires enough VRAM for the MACE-MP-0 medium model and the system size. Reduce layer sizes if you hit OOM errors.

---

## Roadmap

- [ ] Add a minimal C++/CUDA neighbor-list micro-benchmark to compare ASE/PyTorch Geometric neighbor builds with hand-rolled cell/Verlet lists.
- [ ] Provide a LAMMPS input script template that re-imports `mantel_multilayer.lmp` with EAM/ReaxFF + Ewald/PPPM.
- [ ] Add DCD/XTC writer for direct comparison with GROMACS/AMBER/OpenMM trajectories.
- [ ] Validate 316 steel substitution against an experimental SQS or cluster-expansion structure.
- [ ] Extend NIM calculator to batch multiple frames per POST call for throughput-sensitive active-learning (ALCHEMI-style) loops.

---

## Author context / trajectory

This notebook sits at the intersection of **GPU/HPC engineering**, **molecular simulation**, and **ML model serving**. It reflects work across:

- **CUDA/C++ HPC infrastructure** for numerical and ML workloads.
- **MD-engine concepts**: PBC, neighbor lists, force-field evaluation, fixed-atom constraints, and long-range electrostatics (Ewald/PME/PPPM).
- **MLIPs / MACE / MACE-MP-0** and ALCHEMI-style active-learning workflows.
- **NVIDIA BioNeMo / NIM** model hosting and REST-based chemistry/physics microservices.

Timeline (example template — fill in your own dates):

```text
2016–2019: Built and maintained GPU-accelerated HPC/MD pipelines; deep exposure 
            to OpenMM/GROMACS/AMBER integration, CUDA profiling, and MPI scheduling.
2019–2021: Led engineering for a shipped CUDA/C++ numerical library used in 
            production molecular-modeling workflows.
2022–present: Focused on machine-learned interatomic potentials (MACE) and 
              NVIDIA NIM/BioNeMo serving; bridge work between classical MD 
              engines and neural network force fields.
```

If you are a hiring manager or recruiter, the relevant keywords are:** GPU/HPC, CUDA, C++, force fields, neighbor lists, periodic boundary conditions, electrostatics, PME/Ewald, OpenMM/AMBER/GROMACS/LAMMPS, MACE-MP-0, BioNeMo, NVIDIA NIM, ASE, OVITO.

---

## References

- [ASE – Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
- [MACE – Machine Learning Interatomic Potentials](https://github.com/ACEsuit/mace)
- [MACE-MP-0 foundation model](https://github.com/ACEsuit/mace-mp)
- [NVIDIA NIM / BioNeMo](https://www.nvidia.com/en-us/ai/bionemo/)
- [OVITO](https://www.ovito.org/)
- [FIRE optimizer](https://doi.org/10.1103/PhysRevLett.97.170201)

---

**Maintainer note**: Replace the placeholder dates in the *Author context / trajectory* section with your actual timeline. The technical sections are written to match the notebook as-is and can be extended once the C++/CUDA neighbor-list micro-benchmark and LAMMPS input templates are added.
