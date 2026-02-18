"""
Simulación de un gas bidimensional de partículas puntuales en un recinto cuadrado.

El sistema modela un conjunto de partículas de masa unitaria que se mueven
libremente dentro de una caja de lado L, interactuando mediante colisiones
elásticas entre sí y con las paredes del recinto.

Las velocidades iniciales se asignan a partir de una distribución exponencial
asociada a una temperatura inicial T0. Durante la simulación se calcula la
evolución temporal de la presión ejercida sobre las paredes y, al finalizar,
se analiza la distribución de energías cinéticas del sistema.

El objetivo es comprobar numéricamente la conservación de la energía,
estimar la temperatura final del sistema y contrastar la relación del gas
ideal en dos dimensiones.

Autor: Miguel Enterría Lastra
Fecha: 09/02/2026
"""


import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree

# Variables físicas

L = 10  # lado de la caja
mass = 1  # masa del disco
number_particles = 10

kB = 0.01 # Constante de Boltzman
T0 = 50 # Temperatura inicial


# Variables de simulación

dt = 0.01
n_steps = 10000

r_choque = 0.25
r_choque2 = r_choque * r_choque 


# Variables para datos de presión
Presion = 0
tiempo = []
presion_tiempo = []
paso_medida_presion = 100


# Funciones

def energyToSpeed(energy):
    '''
        Función de conversión Energía-Velocidad
    '''
    Speed = np.sqrt(2*energy/mass)
    return Speed


def choque(p1, p2, n):
    '''
        Función que gestiona el evento de un choque entre dos partículas

        Parámetros: 
            - p1, p2: Índices de las dos partículas que intervienen en el choque
            - n: vector normal en la dirección del choque
    '''
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

def energyToTemperature(energy):
    '''
        Conversión de Energía a Temepratura.
    '''
    Temp = 0
    for i in range(number_particles):
        Temp += energy[i]
    Temp /= kB
    Temp /= number_particles
    return Temp

# Cálculo de valores iniciales

angle = 2*m.pi*np.random.rand(number_particles)  # dirección inicial aleatoria
pos = L*np.random.rand(number_particles, 2)      # posición inicial aleatoria (dentro de la caja)
vel = np.zeros((number_particles, 2))
momento = np.zeros(number_particles)

Energia_inicial = np.random.exponential(kB*T0, size=number_particles)
vel[:, 0] = energyToSpeed(Energia_inicial) * np.cos(angle)
vel[:, 1] = energyToSpeed(Energia_inicial) * np.sin(angle)

E_total_inicial = 0
for i in range(number_particles):
    E_total_inicial += Energia_inicial[i]
print(f"Energía inicial: {E_total_inicial}")


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

    # Rebote con paredes
    for i in range(number_particles):
        if pos[i][0] < 0:
            pos[i][0] = 0  # Estas líneas evitan que la partícula se quede atrapada en los bordes
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
    
    # Cálculo de presión cada cierto número de pasos
    if step % paso_medida_presion == paso_medida_presion-1:
        Presion /= (4*L*paso_medida_presion*dt)  # 4 paredes de longitud L
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


# Cálculos posteriores a la simulación

Energia_final = np.zeros(number_particles)
speeds2 = vel[:,0]**2 + vel[:,1]**2

Energia_final = 0.5 * mass * speeds2
E_total_final = 0
for i in range(number_particles):
    E_total_final += Energia_final[i]
print(f"Energía final: {E_total_final}")

print(f"El cambio en energía es: {E_total_final-E_total_inicial}") # Debería ser 0 o un número muy bajo

# Calcular la temperatura del sistema
T_final = energyToTemperature(Energia_final)
print(f"Temperatura final: {T_final}")

# Calcular presión promedio
presion_promedio = np.mean(presion_tiempo)
print(f"Presión promedio: {presion_promedio:.4f}")

# Relación gases ideales
relacion = presion_promedio*L*L/(number_particles*kB*T_final)
print(f"Relación:{relacion}")

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Gráfico de presión vs tiempo
ax1.plot(tiempo, presion_tiempo, 'b-', linewidth=1, label='Presión instantánea')
ax1.axhline(y=presion_promedio, color='r', linestyle='-', linewidth=2, label=f'Presión promedio: {presion_promedio:.4f}')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Presión')
ax1.set_title('Presión en función del tiempo')
ax1.legend()
ax1.grid(True)

# Subplot 2: Histograma de distribución de energía
counts, bins, patches = ax2.hist(
    Energia_final,
    bins='auto',
    edgecolor='black'
)

x_range = bins.max() - bins.min()
margin = 0.2 * x_range
ax2.set_xlim(bins.min() - margin, bins.max() + margin)

ax2.set_xlabel('Energía')
ax2.set_ylabel('Número de partículas')
ax2.set_title('Distribución de Energía')


plt.tight_layout()
plt.show()
