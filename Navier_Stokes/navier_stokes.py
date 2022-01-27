from firedrake import *
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np
import matplotlib.pyplot as plt

class NS_simulation():
    def __init__(self):
        
        self.mesh = Mesh("Mesh/symmetric_mesh.msh")
        coordinates = np.loadtxt("solution/Re_200/mesh_final.txt", delimiter=',')
        self.mesh.coordinates.dat.data[:,:] = coordinates
        
        Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        Vp = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([Vu, Vp])
        
        self.Re = Constant(0.0)
        self.sol = Function(self.V)
        self.sol_test = TestFunction(self.V)
        self.pvd = File("NS_200/sol.pvd")
        
        self.solver_params = {
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
        
        # Get Boundary conditions
        self.boundary_conditions()
        
    def residual(self, sol, sol_test):
        f = Constant((0,0))
        (u,p) = split(sol)
        (v,q) = split(sol_test)

        F = -(
              (1/self.Re)*inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            - inner(f,v)*dx
            + q*div(u)*dx
            )
        return F

    def boundary_conditions(self):
        self.bcs = [DirichletBC(self.V.sub(0), Constant((1,0)), (10,12)),
                    DirichletBC(self.V.sub(0), Constant((0,0)), (13))]
        
    def solve(self, Re):
        self.Re.assign(Constant(Re))
        F = self.residual(self.sol, self.sol_test)
        solve(F == 0, self.sol, bcs=self.bcs, solver_parameters=self.solver_params)
    
    def save_solution(self):
        u = self.sol.split()[0]
        p = self.sol.split()[1]
        u.rename("u", "u")
        p.rename("p", "p")
        self.pvd.write(u,p)
    
    def transient(self):
        
        #Delta_t = 1/12
        #Tfinal = 60.0
        Delta_t = 1/48
        Tfinal = 15.0
        List_t = np.arange(Delta_t,Tfinal+Delta_t,Delta_t)
        
        self.sol_n = Function(self.V)
        (u_n,_) = split(self.sol_n)
        (u,_) = split(self.sol)
        (v,_) = split(self.sol_test)
        F = (1/Delta_t)*inner(u-u_n,v)*dx - 0.5*(self.residual(self.sol, self.sol_test) + self.residual(self.sol_n, self.sol_test))
        F_prob = NonlinearVariationalProblem(F, self.sol, bcs = self.bcs)
        F_solver = NonlinearVariationalSolver(F_prob, solver_parameters=self.solver_params)
        
        # Solve state equation and compute stability
        #self.solve(Constant(46.228486))
        self.solve(Constant(196.933))
        #self.solve(Constant(20))
        self.stability()
        sol_pert = self.eigenfunctions_R[1]
        norm_pert = norm(sol_pert.split()[0])
        norm_sol = norm(self.sol.split()[0])
        #self.sol_n.assign(self.sol + 0.1*sol_pert*(norm_sol/norm_pert)) # 46.25
        self.sol_n.assign(self.sol + 0.05*sol_pert*(norm_sol/norm_pert))
        self.sol.assign(self.sol_n)
        
        self.save_solution()
        
        for t in List_t:
            print("T = %.2e"%t)
            F_solver.solve()
            self.save_solution()
            self.sol_n.assign(self.sol)
            
    def stability(self):
        trial = TrialFunction(self.V)
        (u_t,p_t) = split(trial)
        (v,q) = split(self.sol_test)
        F = self.residual(self.sol, self.sol_test)
        J = derivative(F, self.sol, trial)
        A = assemble(J, bcs=self.bcs, mat_type="aij")
        massform = inner(u_t, v)*dx
        M = assemble(massform, bcs=self.bcs, mat_type="aij")
        
        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        for bc in self.bcs:
            # Ensure symmetry of M
            M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)
        
        # Number of eigenvalues to try
        num_eigenvalues = 3
        
        # Solver options
        opts = PETSc.Options()
        
        parameters = {
             "mat_type": "aij",
             "eps_monitor_conv" : None,
             "eps_converged_reason": None,
             "eps_type": "krylovschur",
             "eps_nev" : num_eigenvalues,
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
        
        # Solve the eigenvalue problem
        eps = SLEPc.EPS()
        eps.create(comm=COMM_WORLD)
        eps.setOperators(A.M.handle, M.M.handle)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setFromOptions()
        eps.setTarget(0)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) 
        #eps.BVSetMatrix(M.M.handle)
        print("### Solving eigenvalue problem ###")
        eps.solve()
        
        # Save eigenvalues
        eigenvalues = []
        self.eigenfunctions_R = []
        self.eigenfunctions_I = []
        ev_re, ev_im = A.M.handle.getVecs()
        
        for i in range(eps.getConverged()):
            eigenvalues.append(eps.getEigenvalue(i))
            eps.getEigenpair(i, ev_re, ev_im)
            # Save the real part
            eigenfunction = Function(self.V, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_re)
            self.eigenfunctions_R.append(eigenfunction)
            # Save the imaginary part
            eigenfunction = Function(self.V, name="Eigenfunction")
            eigenfunction.vector().set_local(ev_im)
            self.eigenfunctions_I.append(eigenfunction)
        
        
        print(norm(self.eigenfunctions_R[1].split()[0]))
        print(norm(self.eigenfunctions_I[1].split()[0]))
            
        return eigenvalues
    
    def compute_stability(self):
        # Critical = 45.4
        Re_array = np.linspace(1,100,100)
        Re_list = []
        eig_R_list = []
        eig_I_list = []
        for Re in Re_array:
            
            print("### Re = %f ###\n" % Re)
            
            NS.solve(Re)
            eigenvalues = NS.stability()
            eig_R = [eig.real for eig in eigenvalues]
            eig_I = [eig.imag for eig in eigenvalues]
            
            Re_list += [Re for i in range(len(eigenvalues))]
            eig_R_list += eig_R
            eig_I_list += eig_I
        
            eig_R_save = np.vstack((np.array(Re_list),np.array(eig_R_list))).transpose()
            np.savetxt("Stability/eig_real.csv",eig_R_save,delimiter=',')
            eig_I_save = np.vstack((np.array(Re_list),np.array(eig_I_list))).transpose()
            np.savetxt("Stability/eig_imag.csv",eig_I_save,delimiter=',')
            
            self.plot_stability()
            
    def plot_stability(self):
        eig_R = np.genfromtxt('Stability/eig_real.csv', delimiter=',')
        eig_I = np.genfromtxt('Stability/eig_imag.csv', delimiter=',')
        
        plt.close("all")
        fig = plt.figure(figsize=(8, 2.5))
        plt.subplot(1,2,1)
        plt.plot(eig_R[:,0],eig_R[:,1],'.')
        plt.axhline(0, xmin=0, xmax=1, color="r", linewidth=0.5)
        plt.title("Real")
        plt.xlabel("Re")
        
        plt.subplot(1,2,2)
        plt.plot(eig_I[:,0],eig_I[:,1],'.')
        plt.ylim(bottom=0)
        plt.title("Imaginary")
        plt.xlabel("Re")
        plt.savefig("stability.png",dpi=600,bbox_inches="tight")
        
if __name__ == "__main__":
    
    NS = NS_simulation()
    #NS.solve(46.25)
    #NS.solve(50)
    NS.transient()
    #NS.save_solution()
    #NS.stability()
    
    
    #NS.compute_stability()
    #NS.plot_stability()