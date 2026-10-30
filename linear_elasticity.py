import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

L = 1.0
W = 0.2
mu = 1.0
rho = 1.0
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([L,W,W])],     # opposite corners of the box
    [20,6,6],                                   # divides those three directions into 20, 6, and 6 cells. That makes \(20\times6\times6=720\) cells total.
    cell_type=mesh.CellType.hexahedron,
)

# Defining the function space
# fem.functionspace(domain, ...) —creates that space on the mesh called domain. 
# "Lagrange" — chooses Lagrange finite elements, which represent a field through its values at mesh points.
# 1 — chooses degree-one elements: the displacement varies linearly within each element.
# (domain.geometry.dim,) — gives the field one component per spatial direction. Since the mesh is 3D, this means three displacement components, \(u_x,u_y,u_z\).
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


# Boundary conditions
def clamped_boundary(x):
    # returns True for points whose \(x\)-coordinate is approximately zero. Those points lie on the face \(x=0\).
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1 # gets the dimension of a face. The mesh is 3D, so its faces are 2D; therefore fdim is 2.
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary) # finds boundary faces matching the function we defined. boundary_facets stores those face identifiers.

u_D = np.array([0, 0, 0], dtype=default_scalar_type) # specifies the displacement at the clamped face: zero in all three directions.
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

ds = ufl.Measure("ds", domain=domain)


# Variational formulation
def epsilon(u):
    return ufl.sym(
        ufl.grad(u)
    )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds


# Solve the linear variational problem
problem = LinearProblem(
    a,
    L,
    bcs=[bc],       # applies the fixed-displacement boundary condition. It is given as a list because a problem can have several boundary conditions.
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, # sets options for PETSc, the solver library used here.- "ksp_type": "preonly" tells PETSc to apply the chosen solver once, without an outer iterative Krylov method.
                                                            # - "pc_type": "lu" chooses LU factorization, a direct method for solving the linear system.
    petsc_options_prefix="linear_elasticity",   # gives this problem’s PETSc options a unique prefix.
)
uh = problem.solve()


# Visualisation
# Create pyvista grid
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

root = 0
assert root < domain.comm.size, f"Cannot gather on process {root} of {domain.comm.size}"

pieces = domain.comm.gather(grid, root=root)

if pieces is not None:
    merged_grid = pyvista.merge(pieces)
    p = pyvista.Plotter()
    actor_0 = p.add_mesh(merged_grid, style="wireframe", color="k")
    actor_1 = p.add_mesh(merged_grid.warp_by_vector("u", factor=1.5), show_edges=True)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("deflection.png")

with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)

# Stress computation
s = sigma(uh) - 1.0 / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

grid.cell_data["VonMises"] = stresses.x.petsc_vec.array
grid.set_active_scalars("VonMises")

stress_clim = [
    domain.comm.reduce(stresses.x.petsc_vec.array.min(), op=MPI.MIN, root=root),
    domain.comm.reduce(stresses.x.petsc_vec.array.max(), op=MPI.MAX, root=root),
]
stress_pieces = domain.comm.gather(grid, root=root)

if stress_pieces is not None:
    merged_stress_grid = pyvista.merge(stress_pieces)
    p = pyvista.Plotter()
    p.add_mesh(merged_stress_grid.warp_by_vector("u", factor=1.5), clim=stress_clim)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show(screenshot="stresses.png")
    else:
        stress_figure = p.screenshot("stresses.png")

