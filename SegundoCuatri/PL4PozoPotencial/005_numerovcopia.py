import numpy as np
import matplotlib.pyplot as plt

def F(k, u, alpha):
    # Potencial adimensional: 0 dentro, k fuera
    return np.where((u > -0.5) & (u < 0.5), -k * alpha, k * (1 - alpha))

def numerov(k, u, alpha, du, psi_prev, psi_curr, i):
    """Implementa un paso del algoritmo de Numerov"""
    du2_12 = (du**2) / 12.0
    f_prev = F(k, u[i-1], alpha)
    f_curr = F(k, u[i], alpha)
    f_next = F(k, u[i+1], alpha)
    
    term_curr = 2.0 * psi_curr * (1.0 + 5.0 * du2_12 * f_curr)
    term_prev = psi_prev * (1.0 - du2_12 * f_prev)
    psi_next = (term_curr - term_prev) / (1.0 - du2_12 * f_next)
    return psi_next

def integracion(k, alpha, u_half, du, paridad_par=True):
    """Integra la ecuación desde u=0 hasta u_max"""
    n = len(u_half)
    psi = np.zeros(n)
    
    if paridad_par:
        psi[0] = 1.0 # psi(0) para estados pares
        # Segundo punto usando expansión de Taylor (derivada nula)
        psi[1] = psi[0] * (1.0 + 0.5 * (du**2) * F(k, u_half[0], alpha))
    else:
        psi[0] = 0.0 # psi(0) para estados impares
        psi[1] = du  # Pendiente inicial no nula
        
    for i in range(1, n - 1):
        psi[i+1] = numerov(k, u_half, alpha, du, psi[i-1], psi[i], i)
        # Control de divergencia extrema
        if abs(psi[i+1]) > 1e6:
            psi[i+1:] = psi[i+1] # Mantener el signo para la bisección
            break
    return psi

def Biseccion_Mejorada(k, u_half, du, paridad_par):
    """Busca autovalores escaneando el espectro de alpha"""
    alphas = np.linspace(0.001, 0.999, 500)
    roots = []
    
    last_val = integracion(k, alphas[0], u_half, du, paridad_par)[-1]
    
    for a in alphas[1:]:
        current_val = integracion(k, a, u_half, du, paridad_par)[-1]
        if last_val * current_val < 0:
            # Refinar con bisección
            a_low, a_high = a - (alphas[1]-alphas[0]), a
            for _ in range(40): # 40 iteraciones para alta precisión
                a_mid = (a_low + a_high) / 2
                if integracion(k, a_low, u_half, du, paridad_par)[-1] * \
                   integracion(k, a_mid, u_half, du, paridad_par)[-1] < 0:
                    a_high = a_mid
                else:
                    a_low = a_mid
            roots.append((a_low + a_high) / 2)
        last_val = current_val
    return roots

def construir_onda_completa(k, alpha, u_half, du, is_par):
    """Construye la función de onda en todo el dominio usando simetría."""
    psi_half = integracion(k, alpha, u_half, du, is_par)
    u_full = np.concatenate([-u_half[::-1], u_half[1:]])
    if is_par:
        psi_full = np.concatenate([psi_half[::-1], psi_half[1:]])
    else:
        psi_full = np.concatenate([-psi_half[::-1], psi_half[1:]])

    # Normalizar para visualización
    max_abs = np.max(np.abs(psi_full))
    if max_abs > 0:
        psi_full = psi_full / max_abs
    return u_full, psi_full

def plot_estados(k, u_half, du, alphas, is_par, ulim, titulo):
    """Dibuja una figura con los estados indicados."""
    plt.figure(figsize=(10, 5))
    for alpha in alphas:
        u_full, psi_full = construir_onda_completa(k, alpha, u_half, du, is_par)
        plt.plot(u_full, psi_full, label=f'α={alpha:.5f}')

    plt.axvline(x=-0.5, color='k', linestyle='--')
    plt.axvline(x=0.5, color='k', linestyle='--')
    plt.xlim(-ulim, ulim)
    plt.ylim(-1.1, 1.1)
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    if len(alphas) > 0:
        plt.legend()

if __name__ == '__main__':
    # Constantes físicas
    V0, a_0 = 244, 1e-10 
    h_b, c = 6.582e-16, 299792458
    me = 0.511e6 / (c**2)
    k = (2 * me * (a_0**2) * V0) / (h_b**2)

    # Configuración del espacio (Solo mitad positiva u >= 0)
    ulim = 2.0
    n_pts = 4001
    u_half = np.linspace(0, ulim, n_pts)
    du = u_half[1] - u_half[0]
    print("du = ",du)
    # Cálculo de autovalores
    alphas_pares = Biseccion_Mejorada(k, u_half, du, paridad_par=True)
    alphas_impares = Biseccion_Mejorada(k, u_half, du, paridad_par=False)
    
    print(f"Alphas Pares: {alphas_pares}")
    print(f"Alphas Impares: {alphas_impares}")

    # Graficación: 3 subplots (uno por nivel de energía)
    niveles = [(a, True) for a in alphas_pares] + [(a, False) for a in alphas_impares]
    niveles.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

    for i, ax in enumerate(axes):
        if i < len(niveles):
            alpha, is_par = niveles[i]
            u_full, psi_full = construir_onda_completa(k, alpha, u_half, du, is_par)
            ax.plot(u_full, psi_full, color='tab:blue')
            ax.axvline(x=-0.5, color='k', linestyle='--')
            ax.axvline(x=0.5, color='k', linestyle='--')
            ax.set_xlim(-ulim, ulim)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(r'$\psi(u)$')
            tipo = "Par" if is_par else "Impar"
            ax.set_title(f"Nivel {i+1}: α={alpha:.5f}")
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No hay más niveles encontrados',
                    transform=ax.transAxes, ha='center', va='center')

    axes[-1].set_xlabel('u')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()