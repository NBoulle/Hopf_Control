from firedrake_adjoint import *
from pyadjoint.tape import no_annotations
import numpy as np
from firedrake import *

# Subclass reduced functional to print parameter values during the optimization
class ReducedFunctionalSafe(ReducedFunctional):
    @no_annotations
    def __call__(self, values):
        try:
            func_value = ReducedFunctional.__call__(self, values)
            blocks = self.tape.get_blocks()
            u = blocks[1].get_outputs()[0].checkpoint
            x = u.split()[0].dat.data[0,0]
            y = u.split()[0].dat.data[0,1]
            params_MS = u.split()[1].dat.data[0]
            mu = u.split()[2].dat.data[0]
            a = values[0].dat.data[0]
            print("a = %f: x = %f, y = %f, params_MS = %f, mu = %f" % (a, x, y, params_MS, mu))
        except:
            func_value = np.nan
            
        return func_value
        