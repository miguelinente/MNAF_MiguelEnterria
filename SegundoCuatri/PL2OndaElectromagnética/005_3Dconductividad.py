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
barrera = 200
t = 0 # Tiempo inicial

eps0 = 8.854e-12
eps1 = 1
eps2 = 4
epsilon = np.zeros((ptos-1,ptos-1))
epsilon[:barrera,:] = eps1
epsilon[barrera:,:] = eps2

sigma1 = 0
sigma2 = 4000
sigma = np.zeros((ptos-1,ptos-1))
sigma[:barrera,:] = sigma1
sigma[barrera:,:] = sigma2

ctc = (dt*sigma)/(2*eps0*epsilon)
ca = (1-ctc)/(1+ctc)
cb = 1/(2*epsilon*(1+ctc))


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

    Ez[1:,1:] =ca*Ez[1:,1:]+ cb*(Hy[1:,1:] - Hy[:-1,1:]) - cb*(Hx[1:,1:] - Hx[1:,:-1])

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
        ax.axvline(x=barrera*dx*1e6, linewidth=2)
        ax2.contourf(XX * 1e6, YY * 1e6, np.clip(Ez, -0.1, 0.1), maplevels, cmap="copper")
        ax2.axvline(x=barrera*dx*1e6, linewidth=2)
        ax3.contourf(XX * 1e6, YY * 1e6, np.clip(Ez, -0.1, 0.1), maplevels, cmap="inferno")
        ax3.axvline(x=barrera*dx*1e6, linewidth=2)
        plt.pause(tpausa)

    t += dt