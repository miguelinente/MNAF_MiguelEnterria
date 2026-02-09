'''
    Simular un disco dentro de una caja rebotando en las paredes

    u.a.
'''

import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree

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

momento = np.zeros(number_particles)

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
n_steps = 10000

# Arrays para almacenar datos de presión
tiempo = np.zeros(n_steps)
presion_tiempo = np.zeros(n_steps)

# Parámetros de colisión
r_choque = 0.25
r_choque2 = r_choque * r_choque 

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

Presion = 0
tiempo = []
presion_tiempo = []
paso_medida = 100

for step in range(n_steps):
    # Avance de posiciones
    pos += vel * dt

    # Rebote con paredes (antes de construir celdas, para no meter índices fuera)
    for i in range(number_particles):
        if pos[i][0] < 0:
            pos[i][0] = 0
            Presion += 2*mass*abs(vel[i][0])
            vel[i][0] = -vel[i][0]
        elif pos[i][0] > L:
            pos[i][0] = L
            Presion += 2*mass*abs(vel[i][0])
            vel[i][0] = -vel[i][0]

        if pos[i][1] < 0:
            pos[i][1] = 0
            Presion += 2*mass*abs(vel[i][1])
            vel[i][1] = -vel[i][1]
        elif pos[i][1] > L:
            pos[i][1] = L
            Presion += 2*mass*abs(vel[i][1])
            vel[i][1] = -vel[i][1]
    
    # Calcular presión final para este paso
    if step % paso_medida == paso_medida-1:
        Presion /= (4*L*paso_medida*dt)  # 4 paredes de longitud L
        # Almacenar datos
        tiempo.append(step*dt)
        presion_tiempo.append(Presion)
        Presion = 0
    
    # Colisiones usando KDTree
    kdtree = KDTree(pos)
    pairs = kdtree.query_pairs(r_choque)
    
    for i, j in pairs:
        d = pos[i] - pos[j]
        dist = np.linalg.norm(d)
        normal = d / dist
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
# Calcular presión promedio
presion_promedio = np.mean(presion_tiempo)
print(f"Presión promedio: {presion_promedio:.4f}")

relacion = presion_promedio*L*L/(number_particles*kB*T_final)
print(f"Relación:{relacion}")

fig,ax = plt.subplots(figsize=(10,6))

ax.plot(tiempo, presion_tiempo, 'b-', linewidth=1, label='Presión instantánea')
ax.axhline(y=presion_promedio, color='r', linestyle='-', linewidth=2, label=f'Presión promedio: {presion_promedio:.4f}')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Presión')
ax.set_title('Presión en función del tiempo')
ax.legend()
ax.grid(True)
plt.show()


