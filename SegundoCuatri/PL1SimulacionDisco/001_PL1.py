
'''
    Simular un disco dentro de una caja rebotando en las paredes

    u.a.
'''

import numpy as np
import math as m
import matplotlib.pyplot as plt 

L = 10 #lado de la caja
mass = 1 #masa del disco
v0 = 5 #velocidad inicial
angle = 2*m.pi*np.random.rand() #dirección inicial aleatoria
pos = L*np.random.random(size=2) #posición inicial aleatoria (dentro de la caja)


'''Datos de la simulación

    Pasos de la simualción: 0.01
    número de pasos de simulación 10 000
    numero de pasos para representacion
    tiempo de pausa entre representaciones 0.01
'''

dt = 0.01
n = 10000

print(f"Ángulo:{angle}, Posición: {pos}")

# Vector de velocidad
vel = np.array([v0*m.cos(angle),v0*m.sin(angle)])

print(vel)

'''Ecuaciones del movimiento

    M.R.U: p = v*t
'''

# Configuración del plot
fig, ax, = plt.subplots(figsize=(8, 8))
ax.set_title("Simulación de partícula")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
plt.ion()  # Activar modo interactivo

for i in range(n):
    pos += vel*dt

    if pos[0] < 0 or pos[0] > L:    
        vel[0] = -vel[0]

    if pos[1] < 0 or pos[1] > L:
        vel[1] = -vel[1]
    
    # Actualizar plot cada 10 pasos
    if i % 10 == 0:
        ax.cla()
        ax.set_title("Simulación de partícula")
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.scatter(pos[0], pos[1], c='red', s=50)
        plt.pause(0.01)

print(pos, vel)
plt.ioff()  # Desactivar modo interactivo
plt.show()


