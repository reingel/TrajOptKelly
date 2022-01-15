import numpy as np
import matplotlib.pyplot as plt
import casadi
from units import *


np.random.seed(1)

m = 1.0*kg
umax = 20*Newton
dmax = 2.0*meter
d = 1.0*meter

h = 0.05
T = 1.0*sec
nt = int(T/h)
t = np.arange(nt)*h

ns = 2
na = 1

opti = casadi.Opti()

x = opti.variable(ns,nt)
u = opti.variable(na,nt)

def objfunc(x, u):
    J = h/2*casadi.sum2((u[0,:-1]**2 + u[0,1:]**2))
    return J

def fk(xk, uk):
    q1, q2 = xk[0,:].reshape((1,-1)), xk[1,:].reshape((1,-1))
    return casadi.vertcat(q2, uk)

def state_grad(x, u):
    xk, xk1 = x[:,:-1], x[:,1:]
    uk, uk1 = u[0,:-1].reshape((1,-1)), u[0,1:].reshape((1,-1))
    q1, q2 = xk[0,:], xk[1,:]
    g = (xk1 - xk) - h/2*(fk(xk,uk) + fk(xk1,uk1))
    return g


opti.minimize(objfunc(x,u))
opti.subject_to(state_grad(x,u) == np.zeros((ns,nt-1)))
opti.subject_to(x[:,0] == np.zeros(ns))
opti.subject_to(x[:,-1] == np.array([d,0]))
opti.subject_to(-dmax <= x[0,:])
opti.subject_to(x[0,:] <= dmax)
opti.subject_to(-umax <= u)
opti.subject_to(u <= umax)
opti.set_initial(x, np.arange(nt)/nt*np.array([d,0]).reshape(-1,1))
opti.set_initial(u, np.zeros((na,nt)))
opti.solver('ipopt')

sol = opti.solve()

xsol = sol.value(x)
usol = sol.value(u)

err = xsol[1,:] - fk(xsol.reshape(x.shape), usol.reshape(u.shape))[1,:]
err = err.flatten()

xanl = 3*t**2-2*t**3
uanl = 6-12*t

plt.figure(1)

plt.subplot(311)
plt.plot(t,xsol[0,:],t,xanl)
plt.grid(True)
plt.title('position')

plt.subplot(312)
plt.plot(t,xsol[1,:]/deg)
plt.grid(True)
plt.title('angle')

plt.subplot(313)
plt.plot(t,usol,t,uanl)
plt.grid(True)
plt.title('force')

plt.figure(2)
plt.plot(t,err,'.-')
plt.grid(True)
plt.title('error')

plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
plt.show()


