from firedrake import *
from fireshape import PdeConstraint
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np
from pyadjoint.tape import no_annotations

class MooreSpence(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()
        
        # Create function space
        self.mesh = mesh_m
        
        # Function space for residual    
        self.Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Vp = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([self.Vu, self.Vp])
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.Z = MixedFunctionSpace([self.Vu, self.Vp, self.R, self.R, self.Vu, self.Vp, self.Vu, self.Vp])
        
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
        (u, p, Re, mu, v_u, v_p, w_u, w_p) = split(self.solution)
        (v, q, test_Re, test_mu, v_v, v_q, w_v, w_q) = split(self.solution_test)
        
        # Residual of the PDE       
        self.F1 = self.residual(u, p, Re, v, q)
        
        # Define normalization function
        self.cu = Constant((1,1))
        self.cp = Constant(0)
        self.inner_v = Constant(0)
        self.inner_w = Constant(1)
        
        # Define the weak form
        self.F2 = derivative(self.residual(u, p, Re, v_v, v_q), self.solution, as_vector([vx for vx in v_u] + [v_p] + [0, 0, 0, 0, 0, 0, 0, 0])) \
                 + mu*inner(w_u,v_v)*dx
        self.F3 = derivative(self.residual(u, p, Re, w_v, w_q), self.solution, as_vector([wx for wx in w_u] + [w_p]  + [0, 0, 0, 0, 0, 0, 0, 0])) \
                 - mu*inner(v_u,w_v)*dx
        self.F4 = (inner(v_u, self.cu) + inner(v_p, self.cp) - self.inner_v)*test_Re*dx
        self.F5 = (inner(w_u, self.cu) + inner(w_p, self.cp) - self.inner_w)*test_mu*dx
        self.F = self.F1 + self.F2 + self.F3 + self.F4 + self.F5
        
        # Boundary conditions
        self.bcs = self.boundary_conditions(self.Z)
        
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
                                "ksp_max_it": 10,
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "schur",
                                "pc_fieldsplit_schur_fact_type": "full",
                                "pc_fieldsplit_0_fields": "0,1,4,5,6,7",
                                "pc_fieldsplit_1_fields": "2,3",
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
    
    # Residual of the Navier-Stokes equations
    def residual(self, u, p, Re, v, q):
        f = Constant((0,0))
        F = -(
              (1/Re)*inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            - inner(f,v)*dx
            + q*div(u)*dx
            )
        return F
    
    # Boundary conditions
    def boundary_conditions(self, Z):
        bcs = [
               DirichletBC(Z.sub(0), Constant((1,0)), (10,12)),
               DirichletBC(Z.sub(0), Constant((0,0)), (13)),
               DirichletBC(Z.sub(4), Constant((1,0)), (10,12,13)),
               DirichletBC(Z.sub(6), Constant((0,0)), (10,12,13))
               ]
        return bcs
    
    # Print parameters of the equations
    @no_annotations
    def print_lambda(self, sol):
        with sol.sub(2).dat.vec_ro as x:
            param = x.norm()
        with sol.sub(3).dat.vec_ro as x:
            mu = x.norm()
        print("### Re = %f, mu = %f ###\n" % (param, mu))
    
    # Compute initial guess for the generalized Moore-Spence system
    @no_annotations
    def compute_guess(self):
        
        # Get boundary conditions  
        bcs = [DirichletBC(self.V.sub(0), Constant((1,0)), (10,12)),
               DirichletBC(self.V.sub(0), Constant((0,0)), (13))]
        
        # Solver parameters       
        solver_params = {
                        "mat_type": "aij",
                        "snes_monitor": None,
                        "snes_linesearch_type": "basic",
                        "snes_max_it": 100,
                        "snes_atol": 1.0e-9,
                        "snes_rtol": 0.0,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps"
                        }
                
        # Get the residual
        z = Function(self.V)
        w = TestFunction(self.V)
        Re = Constant(46.25)
        (vu,vp) =split(z)
        (wu,wp) = split(w)
        Fsol = self.residual(vu, vp, Re, wu, wp)
        
        # Solve the equation
        solve(Fsol == 0, z, bcs=bcs, solver_parameters=solver_params)
        
        # Save the solution
        pvd = File("result/sol.pvd")
        u = z.split()[0]
        p = z.split()[1]
        u.rename("u", "u")
        p.rename("p", "p")
        pvd.write(u,p)
        
        # Solver options
        opts = PETSc.Options()
        parameters = {
                     "mat_type": "aij",
                     "eps_monitor_conv" : None,
                     "eps_converged_reason": None,
                     "eps_type": "krylovschur",
                     "eps_nev" : 2,
                     "eps_max_it": 200,
                     "eps_tol" : 1e-10,
                     "eps_which": "smallest_magnitude",
                     "st_type": "sinvert",
                     "st_ksp_type": "preonly",
                     "st_pc_type": "lu",
                     "st_pc_factor_mat_solver_type": "mumps",
                     "st_ksp_max_it": 10,
                     }
        for k in parameters:
            opts[k] = parameters[k]
        
        # Define eigenvalue problem
        trial = TrialFunction(self.V)
        (u_t,p_t) = split(trial)
        J = derivative(Fsol, z, trial)
        A = assemble(J, bcs=bcs, mat_type="aij")
        massform = inner(u_t, wu)*dx
        M = assemble(massform, bcs=bcs, mat_type="aij")
        
        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        for bc in bcs:
            # Ensure symmetry of M
            M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)
        
        # Solve the eigenvalue problem
        eps = SLEPc.EPS()
        eps.create(comm=COMM_WORLD)
        eps.setOperators(A.M.handle, M.M.handle)
        eps.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        eps.setFromOptions()
        eps.setTarget(0)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) 
        print("### Solving eigenvalue problem ###")
        eps.solve()
        
        # Save eigenvalues
        eig_index = 1
        eigenvalues = []
        eigenfunctions_R = []
        eigenfunctions_I = []
        ev_re, ev_im = A.M.handle.getVecs()
        
        for i in range(eps.getConverged()):
            eigenvalues.append(eps.getEigenvalue(i))
            eps.getEigenpair(i, ev_re, ev_im)
            # Save the real part
            eigenfunction = Function(self.V, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_re)
            eigenfunctions_R.append(eigenfunction)
            # Save the imaginary part
            eigenfunction = Function(self.V, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_im)
            eigenfunctions_I.append(eigenfunction)
        
        # Select the eigenvalue and eigenvector
        mu_sol = eigenvalues[eig_index].imag
        eig_R = eigenfunctions_R[eig_index]
        eig_I = eigenfunctions_I[eig_index]
        
        # Split the components
        (u_r,p_r) = split(eig_R)
        (u_i,p_i) = split(eig_I)
        
        # Compute the two inner product
        inner_v = assemble(inner(u_r, self.cu)*dx + inner(p_r, self.cp)*dx)
        inner_w = assemble(inner(u_i, self.cu)*dx + inner(p_i, self.cp)*dx)
        
        # Renormalize
        theta = np.arctan2(inner_v, inner_w)
        r = 1 / (inner_v*sin(theta) + inner_w*cos(theta))
        phi_r = Function(self.V).assign(eig_R)
        phi_i = Function(self.V).assign(eig_I)
        eig_R.assign(Constant(r*cos(theta))*phi_r-Constant(r*sin(theta))*phi_i)
        eig_I.assign(Constant(r*sin(theta))*phi_r+Constant(r*cos(theta))*phi_i)

        # Save the two eigenfunctions
        eig_R_u = eig_R.split()[0]
        eig_R_p = eig_R.split()[1]
        eig_I_u = eig_I.split()[0]
        eig_I_p = eig_I.split()[1]
        eig_R_u.rename("u","u")
        eig_R_p.rename("p","p")
        eig_I_u.rename("u","u")
        eig_I_p.rename("p","p")
        pvd.write(eig_R_u,eig_R_p)
        pvd.write(eig_I_u,eig_I_p)
        
        ### Assign the initial guess
        # Solution
        self.solution.split()[0].assign(z.split()[0])
        self.solution.split()[1].assign(z.split()[1])
        
        # Parameter
        self.solution.split()[2].assign(Re)
        
        # Eigenvalue
        self.solution.split()[3].assign(mu_sol)
        
        # Eigenvector
        self.solution.split()[4].assign(eig_R_u)
        self.solution.split()[5].assign(eig_R_p)
        self.solution.split()[6].assign(eig_I_u)
        self.solution.split()[7].assign(eig_I_p)
    
    @no_annotations
    def print_update(self):
        area = assemble(1.0*dx(self.mesh))
        # Print relative area wrt to initial area
        print("\n### changed area = %f ###" % (area/self.init_area))
        
        test = Function(self.Vp)
        test.rename("mesh")
        self.mesh_pvd.write(test)
    
    # Solve the generalized Moore-Spence system
    def solve(self):
        super().solve()
        
        self.print_update()
        
        try: 
            print("### Solving generalized Moore-Spence system ###")
            solve(self.F == 0, self.solution, bcs=self.bcs, solver_parameters=self.solver_params)
            self.failed_to_solve = False
            
            # Record the last successful PDE step
            self.solution_n.assign(self.solution)
            
        except:
            # assign last successful optimization step
            self.failed_to_solve = True