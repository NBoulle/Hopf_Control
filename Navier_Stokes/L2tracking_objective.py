from firedrake import *
from fireshape import ShapeObjective
from L2tracking_PDEconstraint import MooreSpence
import numpy as np
from pyadjoint.tape import no_annotations

class L2trackingObjective(ShapeObjective):
    """L2 tracking functional for MooreSpence problem."""

    def __init__(self, pde_solver: MooreSpence, Q, pvd_path, results_path, mesh_folder, target, *args, **kwargs):
        super().__init__(Q, *args, **kwargs)
        
        self.pde_solver = pde_solver
        self.u = pde_solver.solution
        self.Q = Q
        self.Vdet = FunctionSpace(Q.mesh_r, "DG", 0)
        self.detDT = Function(self.Vdet)
        self.minVal = np.inf
        
        # Define callback
        self.cb = self.callback
        self.mesh_folder = mesh_folder
        self.pvd_file = File(pvd_path)
        self.results_path = results_path
        open(self.results_path, 'w+').close()
        self.its = 0

        # Target parameter
        self.target = target
        
    def update(self, x, flag, iteration):
        pass
    
    @no_annotations
    def callback(self):
        # Compute value of the functional
        u, p, Re, mu, _, _, _, _ = split(self.pde_solver.sol_opt)
        with self.pde_solver.sol_opt.sub(2).dat.vec_ro as x:
            param = x.norm()
        
        # Write the solution at each step
        u = self.pde_solver.sol_opt.split()[0]
        u.rename("u", "u")
        p = self.pde_solver.sol_opt.split()[1]
        p.rename("p", "p")
        self.pvd_file.write(u, p)
        
        # Save the value of the function to a file
        f1 = open(self.results_path, "a+")
        f1.writelines("%d,%e\n"%(self.its, param))
        f1.close()
        self.its = self.its + 1
    
    @no_annotations
    def value_form(self):
        """
        Evaluate misfit functional.
        """        
        
        u, p, lmbda, mu, _, _, _, _ = split(self.pde_solver.solution)
        u_opt, p_opt, lmbda_opt, mu_opt, _, _, _, _ = split(self.pde_solver.sol_opt)
                
        # Normalize the functional by the area of the domain
        area = assemble(1.0*dx(self.pde_solver.mesh))
        value = (1/self.target**2)*(1/area)*(lmbda - self.target)**2 * dx
        
        # Print useful informations
        self.detDT.interpolate(det(grad(self.Q.T)))
        with self.pde_solver.solution.sub(2).dat.vec_ro as x:
            param = x.norm()
        RelatDiff = norm(u-u_opt) / norm(u)
        print("\nlambda = %e, functional = %e, diff = %e, det = %e" % (param, (param-self.target)**2 / self.target**2, RelatDiff, min(self.detDT.vector())))
        
        value_assemble = assemble(value)
        value_shrink = 10*self.minVal*(1/area)*dx(self.pde_solver.mesh)
        
        # Ensure that we stay on the same branch
        if norm(u-u_opt) > 5e-2*norm(u):
            #value = np.nan * dx(self.pde_solver.mesh)
            value = value_shrink
            value_assemble = float("inf")
            
        # Return nan if the solver has failed
        if self.pde_solver.failed_to_solve:
            print("### Failed to solve ###")
            #value = np.nan * dx(self.pde_solver.mesh)
            value = value_shrink
            value_assemble = float("inf")
            
        # Ensure not self intersection
        if min(self.detDT.vector()) <= 0.0:
            #value = np.nan * dx(self.pde_solver.mesh)
            value = value_shrink
            value_assemble = float("inf")
        
        # Use the last optimization improvement as initial guess
        if value_assemble < self.minVal:
            self.pde_solver.sol_opt.assign(self.pde_solver.solution)
            self.minVal = value_assemble
            # Save mesh
            np.savetxt(self.mesh_folder+"/mesh_final.txt", self.Q.mesh_m.coordinates.dat.data[:,:], delimiter=',')
        else:
            self.pde_solver.solution.assign(self.pde_solver.sol_opt)
        print("value = %e, norm sol = %e\n"%(value_assemble, norm(u)))
            
        return value
