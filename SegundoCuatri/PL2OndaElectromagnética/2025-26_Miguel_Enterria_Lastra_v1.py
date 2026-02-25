import numpy as np
import matplotlib.pyplot as plt

# Datos del espacio
dx = 10e-9
c = 3e8
ptos = 1001
dt = 0.5 * dx/c

mallado = np.linspace(0, (ptos-1)*dx, ptos)

barrera = 500 #Barrera de medios conductivos
eps0 = 8.854e-12
eps1 = 1
eps2 = 4
epsilon = np.zeros(ptos-1)
epsilon[:barrera] = eps1
epsilon[barrera:] = eps2

sigma1 = 0
sigma2 = 4000
sigma = np.zeros(ptos-1)
sigma[:barrera] = sigma1
sigma[barrera:] = sigma2

ctc = (dt*sigma)/(2*eps0*epsilon)
ca = (1-ctc)/(1+ctc)
cb = 1/(2*epsilon*(1+ctc))

Ey = np.zeros(ptos)
Hz = np.zeros(ptos)
cd = np.zeros(ptos-1)
cd[:barrera] = 1/(2*eps1)
cd[barrera:] = 1/(2*eps2)

# simulación
n_steps = 10000
pasos_rep = 10
tpausa = 0.01

# Pulso
E0 = 1 # Intensidad del pulso
dx_pulso = 400e-9 
dt_pulso = dx_pulso/c
t0_pulso = 5*dt_pulso
k0_pulso = 250
t = 0 # Tiempo inicial

# Plot
fig = plt.figure(figsize=(6,6), clear=True)
ax = plt.subplot(111)

ax.set_xlim(mallado[0]*1e6, mallado[-1]*1e6)
ax.set_ylim(-1.5*E0, 1.5*E0)

# Dibujar línea vertical en la posición de la barrera
x_barrera = mallado[barrera]*1e6  # Convertir a micrómetros
ax.axvline(x_barrera, color='black', linewidth=2)

ax.set_xlabel("Posición (µm)")

ax.set_ylabel("Ey")
ax.set_title("Propagación onda electromagnética")

# Creamos objeto Line2D
line_Ey, = ax.plot(mallado*1e6, Ey, color="blue", label="Ey(x,t)")
line_Hz, = ax.plot(mallado*1e6, Hz, color="red", label="Hz(x,t)")


ax.legend(loc=1)


plt.ion()
#plt.show()

# Bucle principal

Etemp = np.zeros((np.round(2*np.sqrt(eps1))).astype(int))
Htemp = np.zeros((np.round(2*np.sqrt(eps2))).astype(int))

for i in range(n_steps):

    Ey[1:] = ca*Ey[1:] - cb*(Hz[1:] - Hz[:-1])
    
    Ey[k0_pulso] += E0*np.exp(-0.5*((t - t0_pulso)/dt_pulso)**2)

    Ey[0] = Etemp[0]
    Etemp[:-1] = Etemp[1:]
    Etemp[-1] = Ey[1]

    Hz[:-1] += - 0.5*(Ey[1:] - Ey[:-1])

    Hz[-1] = Htemp[0]
    Htemp[:-1] = Htemp[1:]
    Htemp[-1] = Hz[-2]

    if i % pasos_rep == 0:
        line_Ey.set_data(mallado*1e6, Ey)
        line_Hz.set_data(mallado*1e6, Hz)
        plt.pause(tpausa)

    t += dt

plt.ioff()