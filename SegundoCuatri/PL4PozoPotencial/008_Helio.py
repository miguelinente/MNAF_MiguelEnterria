import numpy as np
import math as m
import matplotlib.pyplot as plt

def F(L,rho,alpha,Z):
    return L*(L+1)/(rho**2) - 2*Z/rho + alpha

def numerov(L, rho_space, alpha, drho, n,Z):
    drho2 = drho**2
    
    f_array = F(L, rho_space, alpha,Z)
    
    psi_0 = 0.0
    psi_curr = drho**(L+1) 
    
    phi_prev = 0.0
    phi_curr = psi_curr * (1 - drho2 * f_array[1] / 12)
    
    for i in range(1, n - 1):
        f_k = f_array[i]
        
        phi_next = 2 * phi_curr - phi_prev + drho2 * f_k * psi_curr
        
        psi_next = phi_next / (1 - drho2 * f_array[i+1] / 12)
        
        phi_prev = phi_curr
        phi_curr = phi_next
        psi_curr = psi_next
    
    return psi_curr

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

def Biseccion(L,rho_space,n,drho,intervalo,Z,tol=1e-10,max_iter=200):
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
    # Valores iniciales #

    n_pts = 10001
    rho_lim = 100 # Punto más alejado donde se calucla el valor de la función de onda.
    rho_space = np.linspace(1e-7,rho_lim,n_pts) #Evitar así singularidad
    drho = (rho_lim)/(n_pts - 1)

    Z = 2 #Número atómico del helio

    n_alphas = Z/2*1000
    alphas = np.linspace(0.05, Z + 0.05, n_alphas)
    for L in range(2):
        print("\n----- Calculando para L =  -----\n",L)
        psi = [] #Lista con todos los valores de la función de onda en el infinito
        for alpha in alphas:
            psi.append(numerov(L,rho_space,alpha,drho,n_pts,Z))

        psi = np.array(psi)

        intervalos = barridoInicial(psi, alphas)
        #print("Intervalos: ", intervalos)

        Alphas = Biseccion(L,rho_space,n_pts,drho,intervalos, Z)
        for alpha in Alphas:
            print(f"Energía encontrado: {alpha:.6f}, valor: {-13.6*alpha:.4f} eV")


if __name__ == '__main__':
    
    main()
    