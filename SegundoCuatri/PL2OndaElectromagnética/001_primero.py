import numpy as np
import matplotlib.pyplot as plt

# Datos del espacio

dx = 10e-9

c = 3e8

ptos = 1001

dt = 0.5 * dx/c

mallado = np.linspace(0, (ptos-1)*dx, ptos)
Ey = np.zeros(ptos)
Hz = np.zeros(ptos)

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

ax.set_xlabel("Posición (µm)")
ax.set_ylabel("Ey")
ax.set_title("Propagación onda electromagnética")

# Creamos objeto Line2D
line_Ey, = ax.plot(mallado*1e6, Ey, color="blue", label="Ey(x,t)")
line_Hz, = ax.plot(mallado*1e6, Hz, color="red", label="Hz(x,t)")
ax.legend()

plt.ion()
plt.show()

# Bucle principal

Etemp = np.zeros(2)
Htemp = np.zeros(2)

for i in range(n_steps):

    Ey[1:] += - 0.5*(Hz[1:] - Hz[:-1])
    
    Ey[k0_pulso] += E0*np.exp(-0.5*((t - t0_pulso)/dt_pulso)**2)

    Ey[0] = Etemp[0]
    Etemp[0] = Etemp[1]
    Etemp[1] = Ey[1]

    Hz[:-1] += - 0.5*(Ey[1:] - Ey[:-1])

    Hz[-1] = Htemp[0]
    Htemp[0] = Htemp[1] 
    Htemp[1] = Hz[-2]

    

    if i % pasos_rep == 0:
        line_Ey.set_data(mallado*1e6, Ey)
        line_Hz.set_data(mallado*1e6, Hz)
        plt.pause(tpausa)

    t += dt

plt.ioff()