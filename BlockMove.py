import numpy as np
import matplotlib.pyplot as plt
import casadi
from units import *


np.random.seed(1)

m1 = 1.0*kg
m2 = 0.3*kg
l = 0.5*meter
umax = 20*Newton
dmax = 2.0*meter
d = 1.0*meter

h = 0.05
T = 2.0*sec
NT = int(T/h)
t = np.arange(NT)*h

opti = casadi.Opti()

x = opti.variable(4,NT)
u = opti.variable(1,NT)

def objfunc(x, u):
    J = h/2*casadi.sum2((u[0,:-1]**2 + u[0,1:]**2))
    return J

def trans(xk, uk):
    q1, q2, q3, q4 = xk[0,:], xk[1,:], xk[2,:], xk[3,:]
    return casadi.vertcat(
        q3,
        q4,
        (l*m2*casadi.sin(q2)*q3**2 + uk + m2*grav*casadi.cos(q2)*casadi.sin(q2))/(m1 + m2*(1 - casadi.cos(q2)**2)),
        -(l*m2*casadi.cos(q2)*casadi.sin(q2)*q4**2 + uk*casadi.cos(q2) + (m1 + m2)*grav*casadi.sin(q2))/(l*m1 + l*m2*(1-casadi.cos(q2)**2))
    )

def dynamics(x, u):
    xk, xk1 = x[:,:-1], x[:,1:]
    uk, uk1 = u[0,:-1], u[0,1:]
    q1, q2, q3, q4 = xk[0,:], xk[1,:], xk[2,:], xk[3,:]
    g = (xk1 - xk) - h/2*(trans(xk,uk) + trans(xk1,uk1))
    return g


opti.minimize(objfunc(x,u))
opti.subject_to(dynamics(x,u) == np.zeros((4,NT-1)))
opti.subject_to(x[:,0] == np.zeros(4))
opti.subject_to(x[:,-1] == np.array([d,pi,0,0]))
opti.subject_to(-dmax <= x[0,:])
opti.subject_to(x[0,:] <= dmax)
opti.subject_to(-umax <= u)
opti.subject_to(u <= umax)
opti.set_initial(x, np.arange(NT)/NT*np.array([d,pi,0,0]).reshape(-1,1))
opti.set_initial(u, np.zeros((1,NT)))
opti.solver('ipopt')

sol = opti.solve()

xsol = sol.value(x)
usol = sol.value(u)

plt.figure(1)

plt.subplot(311)
plt.plot(t,xsol[0,:])
plt.grid(True)
plt.title('position')

plt.subplot(312)
plt.plot(t,xsol[1,:]/deg)
plt.grid(True)
plt.title('angle')

plt.subplot(313)
plt.plot(t,usol)
plt.grid(True)
plt.title('force')

plt.show()


