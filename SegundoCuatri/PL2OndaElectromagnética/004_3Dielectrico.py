import numpy as np
import matplotlib.pyplot as plt

# Datos del espacio
dx = 10e-9
c = 3e8
ptos = 401
dt = 0.5 * dx/c

x = np.linspace(0, (ptos-1)*dx, ptos)
y = np.linspace(0, (ptos-1)*dx, ptos)

YY,XX = np.meshgrid(y,x)

Ez = np.zeros((ptos, ptos))
Hx = np.zeros((ptos, ptos))
Hy = np.zeros((ptos, ptos))

# simulación
n_steps = 10000
pasos_rep = 10
tpausa = 0.01

# Pulso
E0 = 1 # Intensidad del pulso
dx_pulso = 40e-9 
dt_pulso = dx_pulso/c
t0_pulso = 5*dt_pulso
k0_pulso = 100
l0_pulso = 200
k_dielec = 200
t = 0 # Tiempo inicial

eps1 = 1
eps2 = 4
cd = np.zeros((ptos-1, ptos-1))
cd[:200,:] = 1/(2*eps1)
cd[200:,:] = 1/(2*eps2)

# Representación inicial:
fig = plt.figure(figsize=(6, 6), clear=True)  # Genero mi "lienzo"
ax = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)

# Formato a los ejes
ax.set(
    title="$E_z$",
    xlabel="Eje X ($\\mu$m)",
    ylabel="Eje Y ($\\mu$m)",
)

ax2.set(
    title="$H_x$",
    xlabel="Eje X ($\\mu$m)",
    ylabel="Eje Y ($\\mu$m)",
)

ax3.set(
    title="$H_y$",
    xlabel="Eje X ($\\mu$m)",
    ylabel="Eje Y ($\\mu$m)",
)

# Intervalo de "altura" del mapa de calor
maplevels = np.linspace(-0.1, 0.1, 21)

# Mapas de calor iniciales
heatmap = ax.contourf(
    XX * 1e6,
    YY * 1e6,
    np.clip(Ez, -0.1, 0.1),
    maplevels,
    cmap="viridis",
)
barra = plt.colorbar(heatmap)

heatmap2 = ax2.contourf(
    XX * 1e6,
    YY * 1e6,
    np.clip(Ez, -0.1, 0.1),
    maplevels,
    cmap="copper",
)
barra2 = plt.colorbar(heatmap2)

heatmap3 = ax3.contourf(
    XX * 1e6,
    YY * 1e6,
    np.clip(Ez, -0.1, 0.1),
    maplevels,
    cmap="inferno",
)
barra3 = plt.colorbar(heatmap3)

fig.tight_layout()# Mapas de calor iniciales

Etemp1= np.zeros((ptos,2))
Etemp2= np.zeros((ptos,2))
Etemp3= np.zeros((ptos,2))
Etemp4= np.zeros((ptos,2))

for i in range(n_steps):

    Ez[1:,1:] += cd*(Hy[1:,1:] - Hy[:-1,1:]) - cd*(Hx[1:,1:] - Hx[1:,:-1])

    Ez[k0_pulso, l0_pulso] = E0*np.exp(-0.5*((t - t0_pulso)/dt_pulso)**2)

    Ez[-1,:] = Etemp1[:,0]
    Etemp1[:,0] = Etemp1[:,1]
    Etemp1[:,1] = Ez[-2,:]

    Ez[0,:] = Etemp2[:,0]
    Etemp2[:,0] = Etemp2[:,1]
    Etemp2[:,1] = Ez[1,:]
    
    Ez[:,0] = Etemp3[:,0]
    Etemp3[:,0] = Etemp3[:,1]
    Etemp3[:,1] = Ez[:,1]

    Ez[:,-1] = Etemp4[:,0]
    Etemp4[:,0] = Etemp4[:,1]
    Etemp4[:,1] = Ez[:,-2]
    
    
    Hx[:,:-1] -= 0.5*(Ez[:,1:] - Ez[:,:-1])

    Hy[:-1,:] += 0.5*(Ez[1:,:] - Ez[:-1,:])

    if i % pasos_rep == 0:
        ax.cla()
        ax2.cla()
        ax3.cla()
        # Vuelvo a darles formato a los ejes
        ax.set(
            title="$E_z$",
            xlabel="Eje X ($\\mu$m)",
            ylabel="Eje Y ($\\mu$m)",
        )
        ax2.set(
            title="$H_x$",
            xlabel="Eje X ($\\mu$m)",
            ylabel="Eje Y ($\\mu$m)",
        )
        ax3.set(
            title="$H_y$",
            xlabel="Eje X ($\\mu$m)",
            ylabel="Eje Y ($\\mu$m)",
        )

        ax.contourf(XX * 1e6, YY * 1e6, np.clip(Ez, -0.1, 0.1), maplevels, cmap="viridis")
        ax2.contourf(XX * 1e6, YY * 1e6, np.clip(Ez, -0.1, 0.1), maplevels, cmap="copper")
        ax3.contourf(XX * 1e6, YY * 1e6, np.clip(Ez, -0.1, 0.1), maplevels, cmap="inferno")
        plt.pause(tpausa)

    t += dt