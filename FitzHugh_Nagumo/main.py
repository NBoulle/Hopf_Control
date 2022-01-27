from firedrake import *
from MS_system import MS_constraint
import numpy as np
from ReducedFunctionalSafe import ReducedFunctionalSafe
from firedrake_adjoint import *

def parameter_optimization(target):
    
    # Create the Augmented system
    problem = MS_constraint()
    
    # Compute the initial guess
    problem.compute_guess()
    
    # Solve the forward problem
    problem.solve_MS()
    
    # Get parameters from the problem
    u = problem.u
    V = problem.V
    
    # Minimize over the target paramter mu (the period)
    mu_target = target
    d = project(Constant(mu_target), V)
    J = assemble(inner(u[3]-d, u[3]-d)*dx) / mu_target**2
    
    # Minimize
    controla = Control(problem.a)
    
    # Print functional value
    def eval_cb(j, m):
        print("\n #### Functional = %g #### \n" % (j))
    
    rf = ReducedFunctionalSafe(J, controla)
    
    # ROL optimization parameters
    params_dict = {
                    'Status Test':{'Gradient Tolerance':1e-6,
                                   'Step Tolerance':1e-12,
                                   'Iteration Limit':100},
                    'Step':{'Type':'Trust Region',
                            'Trust Region':{'Initial Radius': -1, #1e-3 #determine initial radius with heuristics: -1
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
                    'General':{'Print Verbosity':0, #set to any number >0 for increased verbosity
                               'Secant':{'Type':'Limited-Memory BFGS', #BFGS-based Hessian-update in trust-region model
                                         'Maximum Storage':10
                                        }
                              }
                    }
    
    # Solve the minimization problem
    MinProblem = MinimizationProblem(rf)
    inner_product = "L2"
    solver = ROLSolver(MinProblem, params_dict, inner_product=inner_product)
    sol = solver.solve()
    
    # Get the control value
    a = sol.dat.data[0]
    print("a = %f" % a)
    
    # Solve the forward problem
    problem.a.assign(a)
    problem.solve_MS()

if __name__ == "__main__":
    """
    Set the target frequency.
    The default settings correspond to reproducing Fig 1 of the paper.
    """
    
    # Select target bifurcation parameter
    mu_target = 0.0157

    # Run the optimization
    parameter_optimization(mu_target)