# 3D Linear Elasticity with FEniCSx

A learning project that solves the deformation of a clamped 3D beam under its own weight using FEniCSx.

The model uses small-strain linear elasticity, a hexahedral mesh, a fixed displacement boundary condition, and a linear solver. The script also writes the displacement to XDMF and calculates and plots von Mises stress.

## Run

Activate an environment with FEniCSx and the required Python packages, then run:

```bash
python linear_elasticity.py