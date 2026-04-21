import numpy as np
import matplotlib.pyplot as plt

def F(L, rho, alpha, Z):
    term_centrifugo = L * (L + 1) / (rho**2)
    f_int = term_centrifugo - 6.0 / rho + 4.0 + alpha 
    f_ext = term_centrifugo - 2.0 / rho + alpha     
    return np.where(rho < 1.0, f_int, f_ext)

def numerov(L, rho_space, alpha, drho, n, Z):
    '''
        Implementación del método de numerov para encontrar las funciones de onda en todo el espacio

        Entrada:
            L, rho_space, alpha, dhro, n: valores para los cálculos del método

        Salida:
            psi_k: Valor de la función de onda en unpunto alejado del núcleo
    '''
    drho2 = drho**2
    f_array = F(L, rho_space, alpha, Z)
    
    u = np.zeros(2) 
    u[0] = 0.0                      
    u[1] = drho**(L+1)             
    
    phi_prev = u[0] * (1 - drho2 * f_array[0] / 12)
    phi_curr = u[1] * (1 - drho2 * f_array[1] / 12)
    
    psi_k = u[1] 
    
    for k in range(1, n - 1):
        phi_next = 2 * phi_curr - phi_prev + drho2 * f_array[k] * psi_k
        
        psi_next = phi_next / (1 - drho2 * f_array[k+1] / 12)
        
        phi_prev = phi_curr
        phi_curr = phi_next
        psi_k = psi_next
        
    return psi_k

def barridoInicial(psi, alphas):
    '''
        Barrido grueso para encontrar intervalos con cambio de signo

        Entrada:
            psi: array con los valores de psi en u_max
            alphas: array que contiene todos los valores de alpha

        Salida:
            intervalos: interavlos de alphas donde hay cambio de signo  
    '''
    intervalos = []
    
    for i in range(len(alphas)-1):
        if psi[i] == 0:
            intervalos.append((alphas[i],alphas[i]))
        elif psi[i] * psi[i+1] < 0:
            intervalos.append((alphas[i],alphas[i+1]))
    return intervalos

def Biseccion(L,rho_space,n,drho,intervalo,Z,tol=1e-12,max_iter=200):
    '''
        Aplica el método de bisección en los intervalos obtenidos por el barrido inicial
        
        Entrada:
            k,u,alpha,n,du,par: valores para el cálculo de psi en u_max para distintos alphas
            intervalo: lista con intervalos del barrido
            tol: tolerancia del método
            max_iter: máximas iteraciones
        
        Salida:
            alphas: lista con las alphas del método
    '''
    alphas = []

    for alpha_L, alpha_R in intervalo:
        
        # Caso especial: intervalo degenerado
        if alpha_L == alpha_R:
            alphas.append(alpha_L)
            continue

        # Sacamos los valores para el intervalo y comprobamos que haya cambio de signo
        fL = numerov(L, rho_space, alpha_L, drho, n, Z)
        fR = numerov(L, rho_space, alpha_R, drho, n, Z)

        if fL * fR > 0:
            raise ValueError("No hay cambio de signo en este intervalo, introducir un intervalo correcto.")

        iter = 0
        while alpha_R - alpha_L > tol and iter < max_iter:
            alpha_M = alpha_L + (alpha_R - alpha_L)/2
            fM = numerov(L, rho_space, alpha_M, drho, n, Z)

            if fL * fM > 0:
                alpha_L = alpha_M
                fL = fM
            else:
                alpha_R = alpha_M
                fR = fM

            iter += 1

        alphas.append((alpha_L + alpha_R)/2)
    
    return alphas

def main():
    rho_lim = 60
    pasos_por_a0 = 1000  # puntos entre 0 y 1
    n_pts = int(rho_lim * pasos_por_a0) + 1
    rho_space = np.linspace(1e-9, rho_lim, n_pts)
    drho = rho_space[1] - rho_space[0]  

    # En Litio el electrón externo está en n=2. Buscamos alphas entre 0 y 1.
    n_alphas = 2000
    alphas = np.linspace(0.1, 1.1, n_alphas) 
    
    for L in range(2):
        print(f"\n----- Calculando para L = {L} -----\n")
        psi_vals = []
        for alpha in alphas:
            psi_vals.append(numerov(L, rho_space, alpha, drho, n_pts, 3))

        intervalos = barridoInicial(np.array(psi_vals), alphas)
        Alphas_finales = Biseccion(L, rho_space, n_pts, drho, intervalos, 3)
        
        for alpha in Alphas_finales:
            print(f"Alpha encontrado: {alpha:.6f}, Energía: {-13.6*alpha:.4f} eV")

if __name__ == '__main__':
    main()
    