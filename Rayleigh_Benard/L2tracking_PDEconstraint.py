from firedrake import *
from fireshape import PdeConstraint
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np
from pyadjoint.tape import no_annotations

# Load file to load initial guesses
import sys
sys.path.append('../utils')
from vtktools import vtu

RB = __import__("rayleigh-benard")

# Enfore pointwise boundary condition on the pressure
class PointwiseBC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        out = np.array([0], dtype=np.int32)
        return out

class Moore_Spence(PdeConstraint):
    """
    Define the generalized Moore-Spence system for computing Hopf bifurcations.
    """
    
    # Initialization
    def __init__(self, mesh_m):
        super().__init__()
        
        # Create function space
        self.mesh = mesh_m
        self.Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Vp = FunctionSpace(self.mesh, "CG", 1)
        self.VT = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([self.Vu, self.Vp, self.VT])
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.Z = MixedFunctionSpace([self.V, self.R, self.R, self.V, self.V])
        
        # Create solution
        self.sol_opt = Function(self.Z)
        self.solution_n = Function(self.Z)
        self.solution = Function(self.Z, name="State")
        self.solution_test = TestFunction(self.Z)
        
        # Initialize some constans
        self.diff = Constant(-1.0)
        self.init_area = assemble(1.0*dx(self.mesh))
        self.failed_to_solve = False
        
        self.mesh_pvd = File("result/mesh.pvd")
        
        # Write generalized Moore-Spence system of equations
        (u, p, T, Ra, mu, v_u, v_p, v_T, w_u, w_p, w_T) = split(self.solution)
        (v, q, S, test_Ra, test_mu, v_v, v_q, v_S, w_v, w_q, w_S) = split(self.solution_test)
        
        # Residual of the PDE       
        self.F1 = self.residual(u, p, T, Ra, v, q, S)
        
        # Define normalization function
        self.cu = Constant((1,1))
        self.cp = Constant(1)
        self.cT = Constant(1)
        self.inner_v = Constant(0)
        self.inner_w = Constant(1)
        
        # Define the weak form
        self.F2 = derivative(self.residual(u, p, T, Ra, v_v, v_q, v_S), self.solution, as_vector([vx for vx in v_u] + [v_p] + [v_T] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) \
                 + mu*inner(w_u,v_v)*dx + mu*inner(w_T,v_S)*dx
        self.F3 = derivative(self.residual(u, p, T, Ra, w_v, w_q, w_S), self.solution, as_vector([wx for wx in w_u] + [w_p] + [w_T] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) \
                 - mu*inner(v_u,w_v)*dx - mu*inner(v_T,w_S)*dx
        self.F4 = (inner(v_u, self.cu) + inner(v_p, self.cp) + inner(v_T, self.cT) - self.inner_v)*test_Ra*dx
        self.F5 = (inner(w_u, self.cu) + inner(w_p, self.cp) + inner(w_T, self.cT) - self.inner_w)*test_mu*dx
        self.F = self.F1 + self.F2 + self.F3 + self.F4 + self.F5
        
        # Boundary conditions
        self.bcs = self.boundary_conditions(self.Z)
        
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
                                "ksp_monitor_true_residual": None,
                                "ksp_max_it": 10,
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "schur",
                                "pc_fieldsplit_schur_fact_type": "full",
                                "pc_fieldsplit_0_fields": "0,1,2,5,6,7,8,9,10",
                                "pc_fieldsplit_1_fields": "3,4",
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
    
    # Boundary conditions
    def boundary_conditions(self, Z):
        bcs = [
                DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                PointwiseBC(Z.sub(1), Constant(0.0), (1,2,3,4)), # p(0,0) = 0
                DirichletBC(Z.sub(2), Constant(1.0), (3)), # T = 1 at y = 0
                DirichletBC(Z.sub(2), Constant(0.0), (4)), # T = 0 at y = 1
                DirichletBC(Z.sub(5), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                PointwiseBC(Z.sub(6), Constant(0.0), (1,2,3,4)), # p(0,0) = 0
                DirichletBC(Z.sub(7), Constant(0.0), (3,4)), # T = 0 at y = 0,1
                DirichletBC(Z.sub(8), Constant((0.0,0.0)), (1, 2, 3, 4)), # u = 0 at the Boundary
                PointwiseBC(Z.sub(9), Constant(0.0), (1,2,3,4)), # p(0,0) = 0
                DirichletBC(Z.sub(10), Constant(0.0), (3,4)) # T = 0 at y = 0,1
                ]
        return bcs
    
    # Residual of the system
    def residual(self, u, p, T, Ra, v, q, S):
        Pr = 1.0
        g = Constant((0, 1))
        nn  = FacetNormal(self.mesh)
        F = -1.0*(
            - inner(grad(u), grad(v))*dx + inner(dot(u,grad(u)),v)*dx + inner(p, div(v))*dx + Ra*Pr*inner(T*g, v)*dx
            + inner(div(u), q)*dx
            - inner(grad(T), grad(S))*dx
            - inner(dot(u, grad(T)), S)*dx + inner(dot(grad(T),nn),S)*ds(3) + inner(dot(grad(T),nn),S)*ds(4)
            )
        return F
    
    # Print bifurcation parameter
    def print_lambda(self, sol):
        with sol.sub(3).dat.vec_ro as x:
            param = x.norm()
        with sol.sub(4).dat.vec_ro as x:
            mu = x.norm()
        print("### Ra = %f, mu = %f ###\n" % (param, mu))
    
    # Compute initial guess for the generalized Moore-Spence system
    @no_annotations
    def compute_guess(self):
        
        self.problem = RB.RayleighBenardProblem()

        # Initial parameters
        eig_index = 1
        
        # Get boundary conditions  
        bcs = self.problem.boundary_conditions(self.V, [])
        
        # Solver parameters       
        solver_params = self.problem.solver_parameters([],[])
                
        # Get the residual
        z = Function(self.V)
        w = TestFunction(self.V)
        Ra = Constant(param)
        
        (u,p,T) = split(z)
        (v,q,S) = split(w)
        Fsol = self.residual(u, p, T, Ra, v, q, S)
        
        # Load the initial guess
        vtu_class = vtu("Initial_guess/solution.vtu")
        
        # Load velocity
        W_guess = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        X_guess = interpolate(SpatialCoordinate(self.mesh), W_guess)
        reader_guess = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "u")[:,0:2]
        u_vtk = Function(self.Vu)
        u_vtk.dat.data[:] = reader_guess(X_guess.dat.data_ro)
        
        # Load pressure
        W_guess = VectorFunctionSpace(self.mesh, "CG", 1, dim=2)
        X_guess = interpolate(SpatialCoordinate(self.mesh), W_guess)
        reader_guess = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "p")[:,0]
        p_vtk = Function(self.Vp)
        p_vtk.dat.data[:] = reader_guess(X_guess.dat.data_ro)
        
        # Load temperature
        reader_guess = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "T")[:,0]
        t_vtk = Function(self.VT)
        t_vtk.dat.data[:] = reader_guess(X_guess.dat.data_ro)

        # Assign initial guess        
        z.split()[0].assign(u_vtk)
        z.split()[1].assign(p_vtk)
        z.split()[2].assign(t_vtk)
        
        # Solve the RB equation
        solve(Fsol == 0, z, bcs=bcs, solver_parameters=solver_params)
        
        # Save the solution
        pvd = File("result/sol.pvd")
        self.problem.save_pvd(z, pvd)
        
        # Solve the eigenvalue problem
        d = self.problem.compute_stability([Ra], [], z)
        
        # Select the eigenvalue and eigenvector
        mu_sol = d["eigenvalues"][eig_index].imag
        eig_R = d["eigenfunctions_R"][eig_index]
        eig_I = d["eigenfunctions_I"][eig_index]
                
        # Split the components
        (u_r,p_r,T_r) = split(eig_R)
        (u_i,p_i,T_i) = split(eig_I)
        
        # Compute the two inner product
        inner_v = assemble(inner(u_r, self.cu)*dx + inner(p_r, self.cp)*dx + inner(T_r, self.cT)*dx)
        inner_w = assemble(inner(u_i, self.cu)*dx + inner(p_i, self.cp)*dx + inner(T_i, self.cT)*dx)
        
        # Renormalize
        theta = np.arctan2(inner_v, inner_w)
        r = 1 / (inner_v*sin(theta) + inner_w*cos(theta))
        phi_r = Function(self.V).assign(eig_R)
        phi_i = Function(self.V).assign(eig_I)
        eig_R.assign(Constant(r*cos(theta))*phi_r-Constant(r*sin(theta))*phi_i)
        eig_I.assign(Constant(r*sin(theta))*phi_r+Constant(r*cos(theta))*phi_i)

        # Save the two eigenfunctions  
        self.problem.save_pvd(eig_R, pvd)
        self.problem.save_pvd(eig_I, pvd)
        
        ### Assign the initial guess
        # Solution
        self.solution.split()[0].assign(z.split()[0])
        self.solution.split()[1].assign(z.split()[1])
        self.solution.split()[2].assign(z.split()[2])
        
        # Parameter
        self.solution.split()[3].assign(Ra)
        
        # Eigenvalue
        self.solution.split()[4].assign(mu_sol)
        
        # Eigenvector
        self.solution.split()[5].assign(eig_R.split()[0])
        self.solution.split()[6].assign(eig_R.split()[1])
        self.solution.split()[7].assign(eig_R.split()[2])
        self.solution.split()[8].assign(eig_I.split()[0])
        self.solution.split()[9].assign(eig_I.split()[1])
        self.solution.split()[10].assign(eig_I.split()[2])
        
    # Solve the generalized Moore-Spence system
    def solve(self):
        super().solve()
                
        area = assemble(1.0*dx(self.mesh))
        # Print relative area wrt to initial area
        print("\n### changed area = %f ###" % (area/self.init_area))
        
        try: 
            print("### Solving Moore-Spence system ###")
            solve(self.F == 0, self.solution, bcs=self.bcs, solver_parameters=self.solver_params)
            self.failed_to_solve = False
            
            # Record the last successful PDE step
            self.solution_n.assign(self.solution)
            
        except:
            # assign last successful optimization step
            self.failed_to_solve = True
