import numpy as np
import matplotlib.pyplot as plt

def F(L, rho, alpha):
    # La singularidad en rho=0 se evita empezando el espacio en un valor pequeño
    return L * (L + 1) / (rho**2) - 2 / rho + alpha

def numerov(L, rho_space, alpha, drho, n):
    '''
        Implementación del método de Numerov hacia adentro (Inward integration).
        Integra desde rho_max hacia el origen.
    '''
    drho2 = drho**2
    f_array = F(L, rho_space, alpha)
    
    # Reservamos espacio para la función reducida U(r)
    u = np.zeros(n)
    
    # Condiciones de contorno en el "infinito" (rho_lim) [cite: 165, 172]
    # Usamos un valor muy pequeño y el decaimiento asintótico esperado: exp(-sqrt(alpha)*rho)
    u[-1] = 1e-15 
    u[-2] = u[-1] * np.exp(np.sqrt(alpha) * drho)
    
    # Definimos phi para Numerov: phi = u * (1 - drho^2 * f / 12) [cite: 83, 157]
    phi_curr = u[-1] * (1 - drho2 * f_array[-1] / 12)
    phi_prev = u[-2] * (1 - drho2 * f_array[-2] / 12)
    
    # Integración hacia atrás (de n-3 hasta 0) 
    # La fórmula es: phi_{k-1} = 2*phi_k - phi_{k+1} + drho^2 * f_k * u_k [cite: 86, 159]
    for i in range(n - 2, 0, -1):
        # u_curr es el valor en el paso i (ya calculado en el paso anterior o inicializado)
        u_curr = u[i]
        
        phi_next = 2 * phi_prev - phi_curr + drho2 * f_array[i] * u_curr
        
        # Actualizamos u[i-1] deshaciendo el cambio de phi [cite: 87, 160]
        u[i-1] = phi_next / (1 - drho2 * f_array[i-1] / 12)
        
        # Desplazamos variables
        phi_curr = phi_prev
        phi_prev = phi_next
        
    # Devolvemos el valor en el origen (u[0]). Buscamos que sea 0.
    return u[0]

def barridoInicial(psi, alphas):
    intervalos = []
    for i in range(len(alphas)-1):
        if psi[i] == 0:
            intervalos.append((alphas[i],alphas[i]))
        elif psi[i] * psi[i+1] < 0:
            intervalos.append((alphas[i],alphas[i+1]))
    return intervalos

def Biseccion(L, rho_space, n, drho, intervalo, tol=1e-12, max_iter=200):
    alphas_encontrados = []
    for alpha_L, alpha_R in intervalo:
        if alpha_L == alpha_R:
            alphas_encontrados.append(alpha_L)
            continue

        fL = numerov(L, rho_space, alpha_L, drho, n)
        fR = numerov(L, rho_space, alpha_R, drho, n)

        if fL * fR > 0: continue

        iter = 0
        while (alpha_R - alpha_L) > tol and iter < max_iter:
            alpha_M = (alpha_L + alpha_R) / 2
            fM = numerov(L, rho_space, alpha_M, drho, n)

            if fL * fM > 0:
                alpha_L = alpha_M
                fL = fM
            else:
                alpha_R = alpha_M
                fR = fM
            iter += 1
        alphas_encontrados.append((alpha_L + alpha_R) / 2)
    return alphas_encontrados

def main():
    # Parámetros de la malla
    n_pts = 40001
    rho_lim = 60 # Reducido para evitar inestabilidad numérica en la exponencial
    rho_space = np.linspace(1e-8, rho_lim, n_pts)
    drho = rho_space[1] - rho_space[0]

    # Rango de alphas para el Hidrógeno (esperamos 1, 0.25, 0.111...)
    n_alphas = 1000
    alphas = np.linspace(0.02, 1.1, n_alphas)
    
    for L in range(2):
        print(f"\n----- Calculando para L = {L} (Inward) -----\n")
        psi_origen = [] 
        for alpha in alphas:
            psi_origen.append(numerov(L, rho_space, alpha, drho, n_pts))

        intervalos = barridoInicial(np.array(psi_origen), alphas)
        Alphas = Biseccion(L, rho_space, n_pts, drho, intervalos)
        
        for alpha in Alphas:
            print(f"Alpha: {alpha:.8f} | Energía: {-13.6*alpha:.6f} eV")

if __name__ == '__main__':
    main()