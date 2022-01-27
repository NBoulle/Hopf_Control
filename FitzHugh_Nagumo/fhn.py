import numpy
from scipy.integrate import solve_ivp

def fhn(t,y, c1, c2):
    a = -0.12
    b = 0.011
    c3 = 0.55
    
    v = y[0]
    w = y[1]

    f0 = c1*v*(v-a)*(1-v) - c2*w
    f1 =  b*(v-c3*w)

    return [f0, f1]

# Initial parameter at the Hopf point
c1 = 0.050417
c2 = 0.05

# Initial parameter far from the Hopf point
c1 = 0.15
c2 = 0.05

# Final parameter at the Hopf point
c1 = 0.050417
c2 = 0.025736

# Final parameter far from the Hopf point
c1 = 0.15
c2 = 0.025736

# initial condition for FHN
y0 = [0.01, 0.0]  # Off into orbit
T = 2000
t = numpy.linspace(0, T, T+1)

sol = solve_ivp(fhn, [0,T], y0, args=(c1,c2), t_eval=t)
print("min_t v = ", min(sol.y[0]))
print("max_t v = ", max(sol.y[0]))

S = numpy.array([[sol.t[i], sol.y[0,i], sol.y[1,i]] for i in range(len(sol.t))])
numpy.savetxt("sol_%.3f_%.3f.csv" %(c1,c2), S, delimiter=',')

# # Plot solutions
import pylab
pylab.figure()
pylab.plot(sol.t, sol.y[0])

pylab.ylabel("v (mV)")
pylab.xlabel("t (s)")
pylab.savefig("fhn_v.png")