from firedrake import *
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
from pyadjoint.tape import no_annotations

class MS_constraint(object):
    
    def __init__(self):
        # Mesh
        mesh = RectangleMesh(200, 100, 2*pi, pi, diagonal="crossed")
        mesh.coordinates.dat.data[:,0] -= pi
        mesh.coordinates.dat.data[:,1] -= pi/2
        
        # Function Spaces
        self.V = FunctionSpace(mesh, "CG", 1)
        self.W = VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
        R = FunctionSpace(mesh, "R", 0)
        Z = MixedFunctionSpace([self.W, R, R, self.W, self.W])
        
        # Define constant nu
        self.nu = Function(R).assign(Constant(1.0))
        
        ### Build the MS augmented system
        self.u = Function(Z)
        (X, params_MS, mu, v, w) = split(self.u)
        tu = TestFunction(Z)
        (tX, tb, tmu, tv, tw) = split(tu)
        
        # Residual of the system
        F1 = self.residual(X, params_MS, tX)
        
        # Normalization function
        x, y = SpatialCoordinate(mesh)
        self.c = Function(self.W).interpolate(as_vector([(x+pi/2)**2+(y+pi)**2,-(x+pi/2)**2-(y+pi)**2]))

        # Define the weak form
        F2 = derivative(self.residual(X, params_MS, tv), self.u, as_vector([v[0], v[1], 0, 0, 0, 0, 0, 0])) + mu*inner(w,tv)*dx
        F3 = derivative(self.residual(X, params_MS, tw), self.u, as_vector([w[0], w[1], 0, 0, 0, 0, 0, 0])) - mu*inner(v,tw)*dx
        F4 = inner(v,self.c)*tb*dx
        F5 = (inner(w,self.c)-Constant(1))*tmu*dx
        self.F = F1 + F2 + F3 + F4 + F5
        
        # Define boundary conditions
        self.bcs = self.boundary_conditions(Z)
        
        # Solver parameters
        self.solver_params = {
                                "mat_type": "matfree",
                                "snes_type": "newtonls",
                                "snes_monitor": None,
                                "snes_converged_reason": None,
                                "snes_linesearch_type": "l2",
                                "snes_linesearch_maxstep": 1.0,
                                "snes_linesearch_damping": 1.0,
                                "snes_max_it": 100,
                                "snes_atol": 1.0e-8,
                                "snes_rtol": 0.0,
                                "snes_stol": 0.0,
                                "ksp_type": "fgmres",
                                "ksp_max_it": 50,
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "schur",
                                "pc_fieldsplit_schur_fact_type": "full",
                                "pc_fieldsplit_0_fields": "0,3,4",
                                "pc_fieldsplit_1_fields": "1,2",
                                "fieldsplit_0_ksp_type": "preonly",
                                "fieldsplit_0_pc_type": "python",
                                "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                                "fieldsplit_0_assembled_pc_type": "lu",
                                "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
                                "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
                                "mat_mumps_icntl_14": 200,
                                "fieldsplit_1_ksp_type": "gmres",
                                "fieldsplit_1_ksp_max_it": 2,
                                "fieldsplit_1_ksp_convergence_test": "skip",
                                "fieldsplit_1_pc_type": "none",
                            }
    
    # Define residual of the system
    def residual(self, u, r, tu):
        (ur, uc) = split(u)
        (tr, tc) = split(tu)
        mu = 0.1
        c3 = -1.0
        c5 = 1.0
        mag = ur**2 + uc**2
        F = - inner(grad(ur),grad(tr))*dx + (r*ur - self.nu*uc - mag*(c3*ur - mu*uc) - c5*mag*ur)*tr*dx \
            - inner(grad(uc),grad(tc))*dx + (r*uc + self.nu*ur - mag*(c3*uc + mu*ur) - c5*mag*uc)*tc*dx
        return F
    
    # Boundary conditions
    def boundary_conditions(self, Z):
        bcs = [
                DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                DirichletBC(Z.sub(3), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                DirichletBC(Z.sub(4), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                ]
        return bcs
    
    # Compute initial guess for the generalized Moore-Spence system
    @no_annotations
    def compute_guess(self):
        
        # Initial bifurcation parameter r
        params_MS = 2.0
        
        # Trivial branch solution
        usol = Function(self.W)
        tu = TestFunction(self.W)
        Fsol = self.residual(usol, params_MS, tu)
        
        # Boundary conditions
        bcs = [DirichletBC(self.W, Constant((0,0)), "on_boundary")]
        
        # Eigensolver parameters
        solver_parameters = {
                            "mat_type": "aij",
                            "snes_max_it": 20,
                            "snes_type": "newtonls",
                            "snes_linesearch_type": "basic",
                            "snes_stol": 0.0,
                            "snes_atol": 1.0e-8,
                            "snes_rtol": 0.0,
                            "snes_divergence_tolerance": -1,
                            "snes_monitor": None,
                            "snes_converged_reason": None,
                            "snes_linesearch_monitor": None,
                            "ksp_type": "preonly",
                            "ksp_max_it": 10,
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "eps_monitor_conv" : None,
                            "eps_converged_reason": None,
                            "eps_type": "krylovschur",
                            "eps_nev" : 10,
                            "eps_max_it": 50,
                            "eps_tol" : 1e-10,
                            "st_type": "sinvert",
                            "st_ksp_type": "preonly",
                            "st_pc_type": "lu",
                            "st_pc_factor_mat_solver_type": "mumps",
                            "st_ksp_max_it": 10,
                            }
        
        # Solver options
        opts = PETSc.Options()
        for k in solver_parameters:
            opts[k] = solver_parameters[k]
        
        # Setup and solve the eigenvalue problem
        eps = SLEPc.EPS().create(comm=COMM_WORLD)
        eps.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        eps.setFromOptions()
        eps.setTarget(0)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) 
        J = derivative(Fsol, usol, TrialFunction(self.W))
        A = assemble(J, bcs=bcs, mat_type="aij")
        M = assemble(inner(TestFunction(self.W), TrialFunction(self.W))*dx, bcs=bcs, mat_type="aij")
        print("### Solving eigenvalue problem ###")
        eps.setOperators(A.M.handle, M.M.handle)
        eps.solve()
        
        # Save the eigenvector and eigenvalue
        index_eig = 0
        mu_sol = eps.getEigenvalue(index_eig).imag
        ev_re, ev_im = A.M.handle.getVecs()
        eps.getEigenpair(index_eig, ev_re, ev_im)
        # Save the real part
        eig_R = Function(self.W, name="Eigenfunction")
        eig_R.vector().set_local(ev_re)

        # Save the imaginary part
        eig_I = Function(self.W, name="Eigenfunction")
        eig_I.vector().set_local(ev_im)
        
        # Compute the two inner product
        inner_v = assemble(inner(eig_R, self.c)*dx)
        inner_w = assemble(inner(eig_I, self.c)*dx)
        
        print("Init normalization : (%f, %f)"%(inner_v,inner_w))
        
        # Renormalize
        theta = np.arctan2(inner_v, inner_w)
        r = 1 / (inner_v*sin(theta) + inner_w*cos(theta))
        phi_r = Function(self.W).assign(eig_R)
        phi_i = Function(self.W).assign(eig_I)
        eig_R.assign(Constant(r*cos(theta))*phi_r-Constant(r*sin(theta))*phi_i)
        eig_I.assign(Constant(r*sin(theta))*phi_r+Constant(r*cos(theta))*phi_i)
        
        # Compute the two inner product
        inner_v = assemble(inner(eig_R, self.c)*dx)
        inner_w = assemble(inner(eig_I, self.c)*dx)
        print("Final normalization : (%f, %f), expected (0,1)"%(inner_v,inner_w))
        
        ### Assign the initial guess
        # x, y
        self.u.split()[0].assign(usol)
        # b
        self.u.split()[1].assign(params_MS)
        # mu
        self.u.split()[2].assign(mu_sol)
        # v
        self.u.split()[3].assign(eig_R)
        # w
        self.u.split()[4].assign(eig_I)
    
    # Print the system's parameters
    @no_annotations
    def print_lambda(self, sol):
        with sol.sub(1).dat.vec_ro as x:
            param = x.norm()
        with sol.sub(2).dat.vec_ro as x:
            mu = x.norm()
        print("### r = %f, mu = %f ###\n" % (param, mu))
    
    # Solve the generalized MS system
    def solve_MS(self):
        solve(self.F == 0, self.u, bcs=self.bcs, solver_parameters=self.solver_params)
        self.print_lambda(self.u)
        
if __name__ == "__main__":
    
    problem = MS_constraint()

    # Compute the initial guess
    problem.compute_guess()
    
    # Solve the forward problem
    problem.solve_MS()
