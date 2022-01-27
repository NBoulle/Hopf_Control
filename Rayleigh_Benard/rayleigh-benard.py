import sys
from math import floor
from petsc4py import PETSc
from firedrake import *
from firedrake.mg.utils import get_level
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy
import numpy as np

# This hack enforces the boundary condition at (0, 0)
class PointwiseBC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        x = self.function_space().mesh().coordinates.dat.data_ro
        zero = numpy.array([0, 0])
        dists = [numpy.linalg.norm(pt - zero) for pt in x]
        minpt = numpy.argmin(dists)
        if dists[minpt] < 1.0e-10:
            out = numpy.array([minpt], dtype=numpy.int32)
        else:
            out = numpy.array([], dtype=numpy.int32)
        return out

class RayleighBenardProblem():
    
    def mesh(self,comm, levels = 1):
        mesh = RectangleMesh(25, 25, 1, 1, comm=comm)
        mh = MeshHierarchy(mesh, 1)
        self.mesh = mh[1]
        return self.mesh

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 2)
        W = FunctionSpace(mesh, "CG", 1)
        T = FunctionSpace(mesh, "CG", 1)
        self.Z = MixedFunctionSpace([V, W, T])
        return self.Z

    def parameters(self):
        Ra = Constant(0)
        return [(Ra, "Ra", r"$\mathrm{Ra}$")]

    def residual(self, z, Ra, w):
        Pr = 1.0
        (u, p, T) = split(z)
        (v, q, S) = split(w)
        g = Constant((0, 1))
        nn  = FacetNormal(z.function_space().mesh())
        F = -1.0*(
            - inner(grad(u), grad(v))*dx + inner(dot(u,grad(u)),v)*dx + inner(p, div(v))*dx + Ra*Pr*inner(T*g, v)*dx
            + inner(div(u), q)*dx
            - inner(grad(T), grad(S))*dx
            - inner(dot(u, grad(T)), S)*dx + inner(dot(grad(T),nn),S)*ds(3) + inner(dot(grad(T),nn),S)*ds(4)
            )
        return F

    def boundary_conditions(self, Z, params):
        bcs = [
                DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                DirichletBC(Z.sub(2), Constant(1.0), (3)), # T = 1 at y = 0
                DirichletBC(Z.sub(2), Constant(0.0), (4)), # T = 0 at y = 1
                PointwiseBC(Z.sub(1), Constant(0.0), (1,2,3,4))
                ]
        return bcs

    def save_pvd(self, z, pvd):
        (u, p, T) = z.split()
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        T.rename("Temperature", "Temperature")
        pvd.write(u, p, T)
    
    def compute_stability(self, params, branchid, z):
                
        Ra  = params[0]
        Pr = 1.0
        (u0, p0, T0) = split(z)
        self.Z = z.function_space()
        mesh = self.Z.mesh()
        trial = TrialFunction(self.Z)
        (u, p, T) = split(trial)
        test = TestFunction(self.Z)
        (v, q, S) = split(test)
        
        g = Constant((0,1))
        nn  = FacetNormal(mesh)
        x = SpatialCoordinate(mesh)
        
        Fsol = self.residual(z, Ra, test)
        stabform = derivative(Fsol, z, trial)
        
        massform = inner(u, v)*dx + inner(T,S)*dx
        stabbcs = self.boundary_conditions(self.Z, params)
        M = assemble(massform, bcs=stabbcs, mat_type="aij")

        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        for bc in stabbcs:
            M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0.0)
        stabmass = M

        comm = self.Z.mesh().comm
        
        A = assemble(stabform, bcs=stabbcs, mat_type="aij")
        
        # Solver options
        opts = PETSc.Options()

        num_eigenvalues = 3
        parameters = {
             "mat_type": "aij",
             "eps_monitor_conv" : None,
             "eps_converged_reason": None,
             "eps_type": "krylovschur",
             "eps_nev" : num_eigenvalues,
             "eps_max_it": 50,
             "eps_tol" : 1e-10,
             "st_type": "sinvert",
             "st_ksp_type": "preonly",
             "st_pc_type": "lu",
             "st_pc_factor_mat_solver_type": "mumps",
             "st_ksp_max_it": 10,
             }

        for k in parameters:
            opts[k] = parameters[k]
        
        # Create the SLEPc eigensolver
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(A.M.handle, stabmass.M.handle)
        eps.setProblemType(eps.ProblemType.GNHEP)
        eps.setFromOptions()
        
        # Set the target to 0 for real part of the eigenvalue
        eps.setTarget(0)
        eps.setWhichEigenpairs(8)
        
        eps.solve()
        eigenvalues = []
        eigenfunctions_R = []
        eigenfunctions_I = []
        ev_re, ev_im = A.M.handle.getVecs()
        
        for i in range(eps.getConverged()):
            eigenvalues.append(eps.getEigenvalue(i))
            eps.getEigenpair(i, ev_re, ev_im)
            # Save the real part
            eigenfunction = Function(self.Z, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_re)
            eigenfunctions_R.append(eigenfunction)
            # Save the imaginary part
            eigenfunction = Function(self.Z, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_im)
            eigenfunctions_I.append(eigenfunction)
        
        d = {"eigenvalues": eigenvalues,
             "eigenfunctions_R": eigenfunctions_R,
             "eigenfunctions_I": eigenfunctions_I
             }

        return d    
    
    def solver_parameters(self, params, *kwargs):
        linesearch = "basic"
        damping = 1.0
        
        lu = {
             "mat_type": "aij",
             "snes_max_it": 100,
             "snes_type": "newtonls",
             "snes_linesearch_type": linesearch,
             "snes_stol": 0.0,
             "snes_atol": 1.0e-8,
             "snes_rtol": 0.0,
             "snes_divergence_tolerance": -1,
             "snes_monitor": None,
             "snes_converged_reason": None,
             "snes_linesearch_monitor": None,
             "ksp_type": "preonly",
             "ksp_monitor_true_residual": None,
             "ksp_max_it": 10,
             "pc_type": "lu",
             "pc_factor_mat_solver_type": "mumps",
             "eps_monitor_conv" : None,
             "eps_converged_reason": None,
             "eps_type": "krylovschur",
             "eps_nev" : 10,
             "eps_max_it": 50,
             "eps_tol" : 1e-10,
             "eps_target": 500,
             "eps_which": "largest_real",
             "st_type": "sinvert",
             "st_ksp_type": "preonly",
             "st_pc_type": "lu",
             "st_pc_factor_mat_solver_type": "mumps",
             "st_ksp_max_it": 10,
             }
        
        return lu
