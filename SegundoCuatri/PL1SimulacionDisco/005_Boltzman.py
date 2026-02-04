'''
    Simular un disco dentro de una caja rebotando en las paredes

    u.a.
'''

import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 10  # lado de la caja
mass = 1  # masa del disco
number_particles = 100

angle = 2*m.pi*np.random.rand(number_particles)  # dirección inicial aleatoria
pos = L*np.random.rand(number_particles, 2)      # posición inicial aleatoria (dentro de la caja)
vel = np.zeros((number_particles, 2))

kB = 0.01
T0 = 50
Energia_inicial = np.random.exponential(kB*T0, size=number_particles)

def energyToSpeed(energy):
    Speed = np.sqrt(2*energy/mass)
    return Speed

vel[:, 0] = energyToSpeed(Energia_inicial) * np.cos(angle)
vel[:, 1] = energyToSpeed(Energia_inicial) * np.sin(angle)


E_total_inicial = 0
for i in range(number_particles):
    E_total_inicial += Energia_inicial[i]
print(f"Energía inicial: {E_total_inicial}")

def choque(p1, p2, n):
    v_rel = vel[p1] - vel[p2]
    if np.dot(v_rel, n) >= 0:
        return
    t = np.array([-n[1], n[0]])
    v1 = vel[p1]
    v2 = vel[p2]
    v1n = np.dot(v1, n)
    v1t = np.dot(v1, t)
    v2n = np.dot(v2, n)
    v2t = np.dot(v2, t)
    vel[p1] = v2n * n + v1t * t
    vel[p2] = v1n * n + v2t * t
    return

# Datos de la simulación
dt = 0.01
n_steps = 100

# Parámetros de colisión / rejilla
r_choque = 0.25
r_choque2 = r_choque * r_choque
cell_size = r_choque             
nx = int(np.ceil(L / cell_size))
vecinos = [(1,0), (0,1), (1,1), (-1,1)] 

# Configuración del plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Simulación de partícula")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')

plt.ion()
discos = []
for i in range(number_particles-1):
    disco = Line2D([pos[i][0]], [pos[i][1]],
                   marker='o', markersize=12,
                   color='pink', markerfacecolor='violet')
    ax.add_line(disco)
    discos.append(disco)
# Una de un color distinto para distinguirla #
disco = Line2D([pos[i][0]], [pos[i][1]],
                   marker='o', markersize=12,
                   color='blue', markerfacecolor='blue')
ax.add_line(disco)
discos.append(disco)
# ------------------------------------------ # 
fig.canvas.draw()
fig.canvas.flush_events()

for step in range(n_steps):
    # Avance de posiciones
    pos += vel * dt

    # Rebote con paredes (antes de construir celdas, para no meter índices fuera)
    for i in range(number_particles):
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

    # Actualización de la rejilla de la rejilla
    cells = [[] for _ in range(nx * nx)]
    cxi = np.clip((pos[:, 0] / cell_size).astype(int), 0, nx - 1)
    cyi = np.clip((pos[:, 1] / cell_size).astype(int), 0, nx - 1)
    for i in range(number_particles):
        cells[cxi[i] + nx * cyi[i]].append(i)

    # Colisiones usando celdas
    for cy in range(nx):
        for cx in range(nx):
            cell = cells[cx + nx*cy]

            # Choques dentro de la misma celda
            if len(cell) > 1:
                for a in range(len(cell) - 1):
                    i = cell[a]
                    for b in range(a + 1, len(cell)):
                        j = cell[b]

                        d = pos[i] - pos[j]
                        dist2 = d[0]*d[0] + d[1]*d[1]
                        if dist2 < r_choque2 and dist2 > 0.0:
                            normal = d / np.sqrt(dist2)
                            choque(i, j, normal)
            # Choques con celdas vecinas
            if len(cell) > 0:
                for dx, dy in vecinos:
                    cx2 = cx + dx
                    cy2 = cy + dy

                    # límites
                    if not (0 <= cx2 < nx and 0 <= cy2 < nx):
                        continue

                    neighbor_cell = cells[cx2 + nx*cy2]
                    # Comprobación de si las celdas vecinas están vacías 
                    if len(neighbor_cell) == 0:
                        continue

                    for i in cell:
                        for j in neighbor_cell:
                            d = pos[i] - pos[j]
                            dist2 = d[0]*d[0] + d[1]*d[1]
                            if dist2 < r_choque2 and dist2 > 0.0:
                                normal = d / np.sqrt(dist2)
                                choque(i, j, normal)
    # Actualizar plot cada 10 pasos
    if step % 10 == 0:
        for i in range(number_particles):
            discos[i].set_data([pos[i][0]], [pos[i][1]])
        fig.canvas.draw_idle()
        plt.pause(0.01)

plt.ioff()

Energia_final = np.zeros(number_particles)
speeds2 = vel[:,0]**2 + vel[:,1]**2
Energia_final = 0.5 * mass * speeds2
E_total_final = 0
for i in range(number_particles):
    E_total_final += Energia_final[i]
print(f"Energía final: {E_total_final}")

print(f"El cambio en energía es: {E_total_final-E_total_inicial}")

def energyToTemperature(energy):
    Temp = 0
    for i in range(number_particles):
        Temp += energy[i]
    Temp /= kB
    Temp /= number_particles
    return Temp
# Calcular la temperatura del sistema
T_final = energyToTemperature(Energia_final)
print(f"Temperatura final: {T_final}")

fig,ax = plt.subplots(figsize=(8,8))

counts, bins, patches = plt.hist(Energia_final, bins='auto', edgecolor='black')
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
