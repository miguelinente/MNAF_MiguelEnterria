'''
    Simular un disco dentro de una caja rebotando en las paredes

    u.a.
'''

import numpy as np
import math as m
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

L = 10 #lado de la caja
mass = 1 #masa del disco
number_particles = 15
v0 = 1 #velocidad inicial
v0 = np.ones(number_particles)*v0
#v0 = 1 + 2*np.random.rand(number_particles)
angle = 2*m.pi*np.random.rand(number_particles) #dirección inicial aleatoria
pos = L*np.random.rand(number_particles,2) #posición inicial aleatoria (dentro de la caja)
vel = np.zeros((number_particles,2))
vel[:,0] = v0 * np.cos(angle)
vel[:,1] = v0 * np.sin(angle)

def choque(p1,p2,n):
    v_rel = vel[p1] - vel[p2]
    if np.dot(v_rel, n) >= 0:
        return
    t = np.array([-n[1],n[0]])
    v1 = vel[p1]
    v2 = vel[p2]
    v1n = np.dot(v1, n)
    v1t = np.dot(v1, t)
    v2n = np.dot(v2, n)
    v2t = np.dot(v2, t)
    vel[p1] = v2n * n + v1t * t
    vel[p2] = v1n * n + v2t * t
    return

'''Datos de la simulación

    Pasos de la simualción: 0.01
    número de pasos de simulación 10 000
    numero de pasos para representacion
    tiempo de pausa entre representaciones 0.01
'''

dt = 0.01
n = 10000

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
discos = []
for i in range(number_particles):
    disco = Line2D([pos[i][0]],[pos[i][1]],marker='o',markersize=15,color='red',markerfacecolor='red')
    ax.add_line(disco)
    discos.append(disco)

fig.canvas.draw()
fig.canvas.flush_events()

for step in range(n):
    pos += vel*dt

    for i in range(number_particles):
        for j in range(number_particles):
            if i != j:
                d= pos[i]-pos[j]
                dist = np.linalg.norm(d)
                if dist < 0.25:
                    n = d/dist
                    choque(i,j,n)

        if pos[i][0] < 0:
            pos[i][0] = 0     
            vel[i][0] = -vel[i][0]
        elif pos[i][0] > L:
            pos[i][0] = L     
            vel[i][0] = -vel[i][0]
        if pos[i][1] < 0: 
            pos[i][1] = 0   
            vel[i][1] = -vel[i][1]
        elif pos[i][1] > L:
            pos[i][1] = L   
            vel[i][1] = -vel[i][1]
    # Actualizar plot cada 10 pasos
    if step % 10 == 0:
        for i in range(number_particles):
            discos[i].set_data([pos[i][0]],[pos[i][1]])
            
        fig.canvas.draw_idle()
        plt.pause(0.01)
plt.ioff()  
'''
Energia = np.zeros(number_particles)
Energia = 0.5*mass*v0

fig,ax = plt.subplots(figsize=(8,8))

counts, bins, patches = plt.hist(Energia, bins='auto', edgecolor='black')
bin_centers = (bins[:-1] + bins[1:]) / 2
ax.set_xticks(bin_centers)
ax.set_xticklabels([f'{x:.2f}' for x in bin_centers])

# Añadir espacio lateral al histograma
x_range = bins.max() - bins.min()
margin = 0.2 * x_range  
ax.set_xlim(bins.min() - margin, bins.max() + margin)

plt.xlabel('Energía')
plt.ylabel('Número de partículas')
plt.title('Distribución de Energía')
plt.show()
'''

