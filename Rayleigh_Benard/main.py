from firedrake import *
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from L2tracking_PDEconstraint import Moore_Spence
from L2tracking_objective import L2trackingObjective
import numpy as np
import datetime

RB = __import__("rayleigh-benard")

def shape_optimization(target):

    def print_lambda(sol):
            with sol.sub(3).dat.vec_ro as x:
                Ra = x.norm()
            with sol.sub(4).dat.vec_ro as x:
                mu = x.norm()
            print("### Ra = %f, mu = %f ###\n" % (Ra, mu))
    
    # Setup problem
    problem = RB.RayleighBenardProblem()
    mesh = problem.mesh(comm=COMM_WORLD)
    Q = fs.FeControlSpace(mesh)
    inner = fs.ElasticityInnerProduct(Q)
    q = fs.ControlVector(Q, inner)
    
    # Setup PDE constraint
    mesh_m = Q.mesh_m
    e = Moore_Spence(mesh_m)
    
    # Compute the initial guess for the generalized Moore-Spence system
    e.compute_guess()
    e.sol_opt.assign(e.solution)
    
    # create PDEconstrained objective functional
    now = datetime.datetime.now()
    pvd_path = "solution/"+now.strftime("%d-%m-%Y-%H_%M_%S")+"/u.pvd"
    results_path = "solution/"+now.strftime("%d-%m-%Y-%H_%M_%S")+"/results.csv"
    J_ = L2trackingObjective(e, Q, pvd_path, results_path, target)
    J = fs.ReducedObjective(J_, e)
    
    # ROL parameters
    params_dict = {
                    'Status Test':{'Gradient Tolerance':1e-6,
                                   'Step Tolerance':1e-12,
                                   'Iteration Limit':100},
                    'Step':{'Type':'Trust Region',
                            'Trust Region':{'Initial Radius': -1,
                                            'Maximum Radius':1e8,
                                            'Subproblem Solver':'Dogleg',
                                            'Radius Growing Rate':2.5,
                                            'Step Acceptance Threshold':0.05,
                                            'Radius Shrinking Threshold':0.05,
                                            'Radius Growing Threshold':0.9,
                                            'Radius Shrinking Rate (Negative rho)':0.0625,
                                            'Radius Shrinking Rate (Positive rho)':0.25,
                                            'Radius Growing Rate':2.5,
                                            'Sufficient Decrease Parameter':1e-4,
                                            'Safeguard Size':100.0,
                                           }
                           },
                    'General':{'Print Verbosity':0,
                               'Secant':{'Type':'Limited-Memory BFGS', #BFGS-based Hessian-update in trust-region model
                                         'Maximum Storage':10
                                        }
                              }
                    }
    
    params = ROL.ParameterList(params_dict, "Parameters")
    opt_problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(opt_problem, params)
    solver.solve()
    
    # Write final bifurcation parameter
    with e.solution_n.sub(3).dat.vec_ro as x:
        param = x.norm()
    print("\nRa = %e"%param)
    
    # Save final mesh
    np.savetxt("optimization_solution/mesh_final.txt", Q.mesh_m.coordinates.dat.data[:,:], delimiter=',')

if __name__ == "__main__":
    """
    Set the target Hopf bifurcation parameter.
    The default settings correspond to reproducing Fig 3 of the paper.
    """
    
    target = 1.25e5
    
    # Run the shape optimization
    shape_optimization(target)