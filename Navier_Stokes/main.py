from firedrake import *
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from L2tracking_PDEconstraint import MooreSpence
from L2tracking_objective import L2trackingObjective
import numpy as np
import datetime

def shape_optimization(target):
    
    def print_lambda(sol):
            with sol.sub(2).dat.vec_ro as x:
                Re = x.norm()
            with sol.sub(3).dat.vec_ro as x:
                mu = x.norm()
            print("### Re = %f, mu = %f ###\n" % (Ra, mu))
    
    # Setup problem
    mesh = Mesh("Mesh/symmetric_mesh.msh")
    Q = fs.FeControlSpace(mesh)
    # Fix outer boundaries of the domain
    inner = fs.ElasticityInnerProduct(Q, fixed_bids=[10,11,12])
    q = fs.ControlVector(Q, inner)
    
    # Setup PDE constraint
    mesh_m = Q.mesh_m
    e = MooreSpence(mesh_m)
    
    # Compute the initial guess for the generalized Moore-Spence system
    e.compute_guess()
    e.sol_opt.assign(e.solution)
    
    # Create PDEconstrained objective functional
    now = datetime.datetime.now()
    folder_name = "Re_%d" % target
    mesh_folder = "solution/" + folder_name
    pvd_path = "solution/" + folder_name + "/u.pvd"
    results_path = "solution/" + folder_name + "/results.csv"
    J_ = L2trackingObjective(e, Q, pvd_path, results_path, mesh_folder, target)
    J = fs.ReducedObjective(J_, e)
    
    # ROL parameters
    params_dict = {
                    'General': {'Print Verbosity':0,
                                'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
                    'Step': {'Type': 'Augmented Lagrangian',
                             'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
                                                       'Print Intermediate Optimization History': True,
                                                       'Subproblem Iteration Limit': 5}},
                    'Status Test': {'Gradient Tolerance': 1e-6,
                                    'Step Tolerance': 1e-12,
                                    'Constraint Tolerance': 1e-2,
                                    'Iteration Limit': 100}
                    }
    
    # Set up volume and barycentric constraints
    vol = fsz.LevelsetFunctional(Constant(1.0), Q)
    (x, y) = SpatialCoordinate(mesh_m)
    baryx = fsz.LevelsetFunctional(x, Q)
    baryy = fsz.LevelsetFunctional(y, Q)
    econ = fs.EqualityConstraint([vol, baryx, baryy])
    emul = ROL.StdVector(3)
    
    # Solve the optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    opt_problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
    solver = ROL.OptimizationSolver(opt_problem, params)
    solver.solve()
    
    # Write final lambda and save the final mesh
    with e.solution_n.sub(2).dat.vec_ro as x:
        param = x.norm()
    print("\nRe = %e"%param)
    
if __name__ == "__main__":
    """
    Set the target Hopf bifurcation parameter.
    The default settings correspond to reproducing Figs 5 and 6 of the paper.
    """
    
    target = 20
    
    # Run the shape optimization
    shape_optimization(target)
