import numpy as np
import casadi
from units import *

np.random.seed(1)

h = 0.1
m1 = 1.0*kg
m2 = 0.3*kg
l = 0.5*meter
umax = 20*Newton
dmax = 2.0*meter
d = 1.0*meter
T = 2.0*sec
NT = int(T/h)

x = casadi.MX.sym('x',(4,NT))
u = casadi.MX.sym('u',(1,NT))

def objfunc(x, u):
    J = h/2*casadi.sum1(u[0,:-1]**2 + u[0,1:]**2)
    return J

def dynamics(x, u):
    xk, xk1 = x[:,:-1], x[:,1:]
    uk, uk1 = u[0,:-1], u[0,1:]
    q1, q2, q3, q4 = xk[0,:], xk[1,:], xk[2,:], xk[3,:]
    g = xk1 - casadi.vertcat(
        q3,
        q4,
        (l*m2*casadi.sin(q2)*q3**2 + uk + m2*grav*casadi.cos(q2)*casadi.sin(q2))/(m1 + m2*(1 - casadi.cos(q2)**2)),
        -(l*m2*casadi.cos(q2)*casadi.sin(q2)*q4**2 + uk*casadi.cos(q2) + (m1 + m2)*grav*casadi.sin(q2))/(l*m1 + l*m2*(1-casadi.cos(q2)**2))
    )*h
    return g

nlp = {
    'x': casadi.vertcat(x,u),
    'f': objfunc(x,u),
    'g': dynamics(x,u)
}

S = casadi.nlpsol('S', 'ipopt', nlp)
print(S)


