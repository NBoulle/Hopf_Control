from firedrake import *
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
from pyadjoint.tape import no_annotations

class MS_constraint(object):
    """
    Define the PDE constraint as a generalized Moore-Spence system.
    """
    
    # Initialize the system
    def __init__(self):
        
        # Create the mesh
        mesh = UnitIntervalMesh(1)
        
        # Define function Spaces
        self.V = FunctionSpace(mesh, "CG", 1)
        self.W = VectorFunctionSpace(mesh, "DG", degree=0, dim=2)
        R = FunctionSpace(mesh, "R", 0)
        Z = MixedFunctionSpace([self.W, R, R, self.W, self.W])
        
        # Define control parameter a
        self.a = Function(R).assign(Constant(0.05))
        
        # Build the MS augmented system
        self.u = Function(Z)
        (X, params_MS, mu, v, w) = split(self.u)
        tu = TestFunction(Z)
        (tX, tb, tmu, tv, tw) = split(tu)
        
        # Residual of the ODE system
        (x,y) = (X[0], X[1])
        (tx,ty) = (tX[0], tX[1])
        F1 = self.residual(x, y, params_MS, tx, ty)
        
        # Split the functions
        (vx,vy) = (v[0], v[1])
        (wx,wy) = (w[0], w[1])
        
        # Split the test functions
        (tvx,tvy) = (tv[0], tv[1])
        (twx,twy) = (tw[0], tw[1])
        
        # Define normalization function
        self.c = Constant((1,1))
        self.inner_v = Constant(0)
        self.inner_w = Constant(1)
        
        # Define the weak form
        F2 = derivative(self.residual(x, y, params_MS, tvx, tvy), self.u, as_vector([vx, vy, 0, 0, 0, 0, 0, 0])) + mu*inner(w,tv)*dx
        F3 = derivative(self.residual(x, y, params_MS, twx, twy), self.u, as_vector([wx, wy, 0, 0, 0, 0, 0, 0])) - mu*inner(v,tw)*dx
        F4 = (inner(v,self.c)-self.inner_v)*tb*dx
        F5 = (inner(w,self.c)-self.inner_w)*tmu*dx
        self.F = F1 + F2 + F3 + F4 + F5
        
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
                                "snes_atol": 1.0e-8, # 8
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
    
    # Residual of the FitzHugh-Nagumo equations   
    def residual(self, x, y, params_MS, tx, ty):
        a = -0.12
        b = 0.011
        c_1 = 0.175
        c_2 = 0.03
        c_3 = 0.55
        
        c_1 = params_MS
        c_2 = self.a
        
        F1 = (c_1*x*(x-a)*(1-x) - c_2*y)*tx*dx \
            + b*(x - c_3*y)*ty*dx
        return F1
    
    @no_annotations
    def compute_guess(self):
        # Compute initial guess
        
        # Initial parameter c2
        params_MS = 0.15
        
        # Solver parameters       
        solver_params = {
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
                        "ksp_monitor_true_residual": None,
                        "ksp_max_it": 10,
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                        }
                    
        # Solve the FitzHugh-Nagumo equations
        usol = Function(self.W)
        tu = TestFunction(self.W)
        (x,y) = split(usol)
        (tx,ty) = split(tu)
        Fsol = self.residual(x, y, params_MS, tx, ty)
        solve(Fsol == 0, usol, solver_parameters=solver_params)
        x = interpolate(usol[0], self.V)
        y = interpolate(usol[1], self.V)
        print("params_MS = %f: x = %f, y = %f" % (params_MS, x(0.5), y(0.5)))
        
        # Parameters of the eigenvalue solver
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
                            "eps_nev" : 5,
                            "eps_max_it": 50,
                            "eps_tol" : 1e-10,
                            "eps_which": "largest_magnitude",
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
        J = derivative(Fsol, usol, TrialFunction(self.W))
        A = assemble(J, bcs=[], mat_type="aij")
        massform = inner(TestFunction(self.W), TrialFunction(self.W))*dx
        M = assemble(massform, bcs=[], mat_type="aij")
        
        print("### Solving eigenvalue problem ###")
        eps.setOperators(A.M.handle, M.M.handle)
        eps.solve()
        
        # Save eigenvalues
        eig_index = 0
        eigenvalues = []
        eigenfunctions_R = []
        eigenfunctions_I = []
        ev_re, ev_im = A.M.handle.getVecs()
        
        for i in range(eps.getConverged()):
            eigenvalues.append(eps.getEigenvalue(i))
            eps.getEigenpair(i, ev_re, ev_im)
            # Save the real part
            eigenfunction = Function(self.W, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_re)
            eigenfunctions_R.append(eigenfunction)
            # Save the imaginary part
            eigenfunction = Function(self.W, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_im)
            eigenfunctions_I.append(eigenfunction)
        
        # Select the eigenvalue and eigenvector
        mu_sol = eigenvalues[eig_index].imag
        eig_R = eigenfunctions_R[eig_index]
        eig_I = eigenfunctions_I[eig_index]
        
        # Compute the two inner product
        inner_v = assemble(inner(eig_R, self.c)*dx)
        inner_w = assemble(inner(eig_I, self.c)*dx)
        
        # Renormalize
        theta = np.arctan2(inner_v, inner_w)
        r = 1 / (inner_v*sin(theta) + inner_w*cos(theta))
        phi_r = Function(self.W).assign(eig_R)
        phi_i = Function(self.W).assign(eig_I)
        eig_R.assign(Constant(r*cos(theta))*phi_r-Constant(r*sin(theta))*phi_i)
        eig_I.assign(Constant(r*sin(theta))*phi_r+Constant(r*cos(theta))*phi_i)
        
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
    
    # Print parameters of the system
    def print_state(self):
        x = interpolate(self.u[0], self.V)
        y = interpolate(self.u[1], self.V)
        params_MS = interpolate(self.u[2], self.V)
        mu = interpolate(self.u[3], self.V)
        print("a = %f: x = %f, y = %f, params_MS = %f, mu = %f" % (self.a.dat.data[:], x(0.5), y(0.5), params_MS(0.5), mu(0.5)))
    
    # Solve the generalized Moore-Spence system
    def solve_MS(self):
        solve(self.F == 0, self.u, solver_parameters=self.solver_params)
        self.print_state()
        
if __name__ == "__main__":
    
    # Define the problem
    problem = MS_constraint()

    # Compute the initial guess
    problem.compute_guess()
    
    # Solve the forward problem
    problem.solve_MS()
