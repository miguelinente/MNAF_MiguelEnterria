import numpy as np
import matplotlib.pyplot as plt

# =========================
# FUNCIONES
# =========================

def norm(re, im, Ax):
    """Calcula la norma de phi"""
    return np.sqrt(np.sum(re**2 + im**2) * Ax)


def normalization(re, im, Ax):
    """Normaliza phi"""
    norma = norm(re, im, Ax)
    return re / norma, im / norma


def D(phi, Ax):
    """Calcula la primera derivada de phi (condiciones periódicas)"""
    der = np.zeros(len(phi))
    der[1:-1] = phi[2:] - phi[:-2]
    der[0] = phi[1] - phi[-1]
    der[-1] = phi[0] - phi[-2]
    return der / (2 * Ax)


def D2(phi, Ax):
    """Calcula la segunda derivada de phi (condiciones periódicas)"""
    der = np.zeros(len(phi))
    der[1:-1] = phi[2:] + phi[:-2] - 2 * phi[1:-1]
    der[0] = phi[1] + phi[-1] - 2 * phi[0]
    der[-1] = phi[-2] + phi[0] - 2 * phi[-1]
    return der / Ax**2


def p_esp(re, im, Ax):
    """Valor esperado del momento"""
    return np.sum(re * D(im, Ax) - im * D(re, Ax)) * Ax


def p_esp2(re, im, Ax):
    """Valor esperado del momento al cuadrado"""
    return -np.sum(re * D2(re, Ax) + im * D2(im, Ax)) * Ax


def esp(re, im, f, Ax):
    """Valor esperado de f"""
    return np.sum((re**2 + im**2) * f) * Ax


def dx(re, im, f, Ax):
    """Incertidumbre de la posición"""
    return np.sqrt(esp(re, im, f**2, Ax) - esp(re, im, f, Ax)**2)


def dp(re, im, Ax):
    """Incertidumbre del momento"""
    return np.sqrt(p_esp2(re, im, Ax) - p_esp(re, im, Ax)**2)


def prod(dx, dp):
    """Producto de incertidumbres"""
    return dx * dp


# =========================
# DATOS DEL SISTEMA
# =========================

xmin, xmax, ptosx = -5, 5, 1001
mallado = np.linspace(xmin, xmax, ptosx)
Ax = (xmax - xmin) / (ptosx - 1)
At = Ax**2 / 2

# Arrays para almacenamiento temporal
norma = np.array([])
time = np.array([])
P = np.array([])
X = np.array([])
P2 = np.array([])
X2 = np.array([])
dX = np.array([])
dP = np.array([])
Heissenberg = np.array([])

t = 0


# =========================
# POTENCIAL
# =========================

omega = 4
V = 0.5 * omega**2 * mallado**2
Vn = V / max(V)


# =========================
# PARÁMETROS SIMULACIÓN
# =========================

niter = 50000
nwait = 50
tpause = 0.01


# =========================
# CONDICIONES INICIALES
# =========================

x0 = 0
sigma = 0.5
k0 = 10

re = np.exp(-0.5 * ((mallado - x0) / sigma)**2) * np.cos(k0 * mallado)
im = np.exp(-0.5 * ((mallado - x0) / sigma)**2) * np.sin(k0 * mallado)

print("Norma sin normalizar:\n", norm(re, im, Ax))
re, im = normalization(re, im, Ax)
print("Norma normalizada:\n", norm(re, im, Ax))


# =========================
# PRIMERA ITERACIÓN MANUAL
# =========================

time = np.append(time, t)
norma = np.append(norma, norm(re, im, Ax))
P = np.append(P, p_esp(re, im, Ax))
X = np.append(X, esp(re, im, mallado, Ax))
P2 = np.append(P2, p_esp2(re, im, Ax))
X2 = np.append(X2, esp(re, im, mallado**2, Ax))
dP = np.append(dP, dp(re, im, Ax))
dX = np.append(dX, dx(re, im, mallado, Ax))
Heissenberg = np.append(Heissenberg, prod(dX[-1], dP[-1]))


# =========================
# REPRESENTACIÓN INICIAL
# =========================

fig = plt.figure(figsize=(6, 6), clear=True)
plt.suptitle("Ecuación de Schrödinger")

ejes1 = plt.subplot(211)
ejes1.set(
    xlim=[mallado[0], mallado[-1]],
    ylabel=r'$\phi(x)$',
    title="Evolución temporal",
    facecolor='whitesmoke'
)

ejes2 = plt.subplot(234)
ejes2.set(xlabel="Tiempo", ylabel="Norma",
          facecolor='whitesmoke', title="Norma")

ejes3 = plt.subplot(235)
ejes3.set(xlabel="Tiempo",
          facecolor='whitesmoke', title="Valores esperados")

ejes4 = plt.subplot(236)
ejes4.set(xlabel="Tiempo",
          facecolor='whitesmoke', title="Incertidumbres")

ejes1.plot(mallado, Vn, c='teal',
           label=r'$V(x)=\frac{1}{2}\omega^2x^2$', ls='dashed')

ejes4.axhline(0.5, c='k', label=r'$\frac{1}{2}$', ls='dashed')

re_plot, = ejes1.plot(mallado, re, c='orchid', label='Re(φ)')
im_plot, = ejes1.plot(mallado, im, c='darkorchid', label='Im(φ)')
abs_plot, = ejes1.plot(mallado, np.sqrt(re**2 + im**2), c='k', label='|φ|')

norm_plot, = ejes2.plot(time, norma, c='purple', label='Norma')
p_plot, = ejes3.plot(time, P, c='royalblue', label='⟨p⟩')
x_plot, = ejes3.plot(time, X, c='navy', label='⟨x⟩')

dx_plot, = ejes4.plot(time, dX, c='teal', label='Δx')
dp_plot, = ejes4.plot(time, dP, c='lightseagreen', label='Δp')
heissenberg_plot, = ejes4.plot(time, Heissenberg,
                               c='skyblue', label='Δx·Δp')

for ax in [ejes1, ejes2, ejes3, ejes4]:
    ax.grid()
    ax.legend()


# =========================
# BUCLE PRINCIPAL
# =========================

for i in range(niter):

    # Propagación (esquema tipo leapfrog)
    im = im + At/4 * D2(re, Ax) - At/2 * V * re
    re = re - At/2 * D2(im, Ax) + At * V * im
    im = im + At/4 * D2(re, Ax) - At/2 * V * re

    # Actualización observables
    time = np.append(time, At * i)
    norma = np.append(norma, norm(re, im, Ax))
    P = np.append(P, p_esp(re, im, Ax))
    X = np.append(X, esp(re, im, mallado, Ax))
    P2 = np.append(P2, p_esp2(re, im, Ax))
    X2 = np.append(X2, esp(re, im, mallado**2, Ax))
    dP = np.append(dP, dp(re, im, Ax))
    dX = np.append(dX, dx(re, im, mallado, Ax))
    Heissenberg = np.append(Heissenberg, prod(dX[-1], dP[-1]))

    # Actualización gráfica
    if i % nwait == 0:
        re_plot.set_ydata(re)
        im_plot.set_ydata(im)
        abs_plot.set_ydata(np.sqrt(re**2 + im**2))

        norm_plot.set_data(time, norma)
        p_plot.set_data(time, P)
        x_plot.set_data(time, X)
        dx_plot.set_data(time, dX)
        dp_plot.set_data(time, dP)
        heissenberg_plot.set_data(time, Heissenberg)

        ejes2.set_xlim(0, time[-1])
        ejes3.set_xlim(0, time[-1])
        ejes4.set_xlim(0, time[-1])

        plt.pause(tpause)

    t += At