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
    der[0] = phi[1] + phi[-1] -2*phi[0]
    der[-1] = phi[-2] + phi[0] -2*phi[-1]
    return der/(dx**2)

def derx(phi,dx):
    der = np.zeros(len(phi))
    der[1:-1] = phi[2:] -phi[:-2]
    der[0] = phi[1] - phi[-1]
    der[-1] = phi[0] -phi[-2]
    return der /(2*dx)

def esp(re,im,f,dx):
    return np.sum((re**2 + im**2)*f)*dx

def esp_p(re,im,dx):
    dre = derx(re,dx)
    dim = derx(im,dx)
    return np.sum(re*dim - im*dre)*dx

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

omega = 4

V = 0.5*omega**2*mallado**2
Vr = V/max(V)
re = np.zeros(n_ptos)
im = np.zeros(n_ptos)
re = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.cos(k0*mallado)
im = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.sin(k0*mallado)

print(f"Norma antes de normalizar: {norm(re,im,dx)}")
re, im = normalizar(re,im,dx)
print(f"Norma después de normalizar: {norm(re,im,dx)}")

norma = []
tiempo = []
espx = []
espp = []

# Precalcular factores constantes
dt_medio = dt/2
dt_cuarto = dt/4
dt_V = dt*V
dt_V_medio = dt_V/2

fig = plt.figure(figsize=(10,10), clear=True)
ax = plt.subplot(2,1,1)
ax2 = plt.subplot(223)
ax3 = plt.subplot(224)

plt.subplots_adjust(hspace=0.4)

ax.set_xlim(mallado[0], mallado[-1])
ax2.set_xlim(0,dt)
ax3.set_xlim(0,dt)

norm_plot=ax2.plot([], [], c='purple', label='Norma')

espx_plot = ax3.plot([], [], c = 'blue', label='<x>')

espp_plot = ax3.plot([], [], c = 'red', label='<p>')

ax.plot(mallado, Vr, c='teal',label=r'$V(x)=\frac{1}{2}\omega^2x^2$', ls='dashed')

ax.set_xlabel("Posición")
ax.set_ylabel("Phi")
ax.set_title("Onda")
ax.legend()

ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Norma")
ax2.set_title("Evolución de la Norma")
ax2.legend()

ax3.set_xlabel("Tiempo")
ax3.set_ylabel("Valor esperado")
ax3.set_title("Evolución del valor esperado")
ax3.legend()

mod = modulo(re,im)

OndaRe, = ax.plot(mallado, re, color="blue", label = 'Re(φ)')    
OndaIm, = ax.plot(mallado, im, color="red", label = 'Im(φ)') 
Onda, = ax.plot(mallado, mod, color="black", label = '|φ|')       

t = 0

plt.ion()

for i in range(n_pasos):

    im = im + dt_cuarto*d2(re,dx) - dt_V_medio*re
    re = re - dt_medio*d2(im,dx) + dt_V*im
    im = im + dt_cuarto*d2(re,dx) - dt_V_medio*re

    # Solo calcular cuando sea necesario mostrar
    if i%n_p_rep == 0:
        mod = modulo(re,im)
        norma.append(norm(re,im,dx))
        tiempo.append(t)
        espx.append(esp(re,im,mallado,dx))
        espp.append(esp_p(re,im,dx))

    if i%n_p_rep == 0:
        OndaRe.set_data(mallado, re)
        OndaIm.set_data(mallado, im)
        Onda.set_data(mallado, mod)

        norm_plot[0].set_data(tiempo, norma)
        ax2.set_ylim(min(norma),max(norma))
        ax2.set_xlim(0,tiempo[-1])

        espx_plot[0].set_data(tiempo,espx)
        espp_plot[0].set_data(tiempo,espp)
        ax3.set_ylim(min(min(espp),min(espx)),max(max(espx),max(espp)))
        ax3.set_xlim(0,tiempo[-1])
        plt.pause(t_pausa)
    
    t += dt
plt.ioff()
