import numpy as np
import matplotlib.pyplot as plt

def norm(re,im,dx):
    n = np.sqrt(np.sum(re**2 + im**2)*dx)
    return n

def normalizar(re,im,dx):
    n = norm(re,im,dx)
    return re/n, im/n

def modulo(re,im):
    return np.sqrt(re**2 + im**2)

def d2(phi,dx):
    der = np.zeros(len(phi))
    der[1:-1] = (phi[2:] + phi[:-2] - 2*phi[1:-1])
    der[0] = phi[1] -2*phi[0]
    der[-1] = phi[-2] -2*phi[-1]
    return der/(dx**2)

x_min = -5
x_max = 5
n_ptos = 1001
mallado = np.linspace(x_min,x_max,n_ptos)
dx = (x_max -x_min)/(n_ptos-1)
dt = (dx**2)/2

# Simulación #

n_pasos = 50000
n_p_rep = 50
t_pausa = 0.01

# Onda inicial #

x0 = 0
sigma = 0.5
k0 = 10

V = np.zeros(n_ptos)
re = np.zeros(n_ptos)
im = np.zeros(n_ptos)
re = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.cos(k0*mallado)
im = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.sin(k0*mallado)

print(f"Norma antes de normalizar: {norm(re,im,dx)}")
re, im = normalizar(re,im,dx)
print(f"Norma después de normalizar: {norm(re,im,dx)}")

norma = []
tiempo = []

fig = plt.figure(figsize=(6,10), clear=True)
ax = plt.subplot(211)
ax2 = plt.subplot(212)
plt.subplots_adjust(hspace=0.4)

ax.set_xlim(mallado[0], mallado[-1])
ax2.set_xlim(0,dt)

norm_plot=ax2.plot(tiempo, norma, c='purple', label='Norma')

ax.set_xlabel("Posición")
ax.set_ylabel("Phi")
ax.set_title("Onda")
ax.legend()

ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Norma")
ax2.set_title("Evolución de la Norma")
ax2.legend()

mod = modulo(re,im)

OndaRe, = ax.plot(mallado, re, color="blue", label = 'Re(φ)')    
OndaIm, = ax.plot(mallado, im, color="red", label = 'Im(φ)') 
Onda, = ax.plot(mallado, mod, color="black", label = '|φ|')       

t = 0

plt.ion()

for i in range(n_pasos):

    re = re - (dt/2)*d2(im,dx) + dt*V*im
    im = im + (dt/2)*d2(re,dx) - dt*V*re 

    mod = modulo(re,im)

    norma.append(norm(re,im,dx))
    tiempo.append(t)

    if i%n_p_rep == 0:
        OndaRe.set_data(mallado, re)
        OndaIm.set_data(mallado, im)
        Onda.set_data(mallado, mod)

        norm_plot[0].set_data(tiempo, norma)
        ax2.set_ylim(min(norma),max(norma))
        ax2.set_xlim(0,tiempo[-1])
        plt.pause(t_pausa)
    
    t += dt
plt.ioff()
