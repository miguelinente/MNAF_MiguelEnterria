import numpy as np
import matplotlib.pyplot as plt

def F(k,u,alpha):
    return np.where((u > -0.5) & (u < 0.5), -k*alpha, k*(1-alpha))

def numerov(k,u,alpha,du,n):
    '''
        Método de Numerov para hallar las disitntas funciones de onda

        Entrada:
            k,u,alpha,du,n: valores para calcular la función de onda

        Salida:
            psi: valor de la función de onda en el infinito
    '''
    
    du2 = du**2
    f = F(k,u[0],alpha)
    prev = 0 # k-1
    temp = 0
    psi = 0.000001 # psi_k (valor muy pequeño proximo a 0)
    act = psi*(1 - du2*f/12) # k
    
    for i in range(n):
        f = F(k,u[i],alpha)
        temp = act
        act = 2*act - prev + du2*f*psi
        psi = act /(1 - du2*f/12)
        prev = temp
    
    return psi

def construirPsi(k,u,alpha,du,n):
    '''
        Función dedicada a construir psi para las alphas hayadas por el método de bisseción

        Entrada:
            k,u,alpha,du,n: valores para calcular la función de onda
        
        Salida:
            psi: función psi en todo el espaci-o
    '''
    psi_tot = []
    du2 = du**2
    f = F(k,u[0],alpha)
    prev = 0 # k-1
    temp = 0
    psi = 0.000001 # psi_k (valor muy pequeño proximo a 0)
    act = psi*(1 - du2*f/12) # k
    
    for i in range(n):
        psi_tot.append(act)
        f = F(k,u[i],alpha)
        temp = act
        act = 2*act - prev + du2*f*psi
        psi = act /(1 - du2*f/12)
        prev = temp

    # Devolver un array con la misma longitud que `u` (no añadir el append extra)
    return np.array(psi_tot)

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

def Biseccion(k,u,n,du,intervalo,tol=1e-10,max_iter=200):
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
        fL = numerov(k, u, alpha_L, du, n)
        fR = numerov(k, u, alpha_R, du, n)

        if fL * fR > 0:
            raise ValueError("No hay cambio de signo en este intervalo, introducir un intervalo correcto.")

        iter = 0
        while alpha_R - alpha_L > tol and iter < max_iter:
            alpha_M = alpha_L + (alpha_R - alpha_L)/2
            fM = numerov(k, u, alpha_M, du, n)

            if fL * fM > 0:
                alpha_L = alpha_M
                fL = fM
            else:
                alpha_R = alpha_M
                fR = fM

            iter += 1

        alphas.append((alpha_L + alpha_R)/2)
    
    return alphas

def calculos():
    # Valores iniciales #

    V0 = 244
    a = 1e-10 
    h_b = 6.582e-16
    c = 299792458
    me = 0.511e6/(c**2)

    k = (2*me*(a**2)*V0)/(h_b**2)

    n_pts = 4001
    ulim = 2 # Punto más alejado donde se calucla el valor de la función de onda.
    u = np.linspace(-ulim,ulim,n_pts)
    du = (ulim*2)/(n_pts - 1)

    n_alphas = 2000
    alphas = np.linspace(0.001, 0.999, n_alphas)

    psi = []
    for alpha in alphas:
        psi.append(numerov(k,u,alpha,du,n_pts))

    psi = np.array(psi)

    intervalos = barridoInicial(psi, alphas)
    print("Intervalos: ", intervalos)

    alphas_biseccion = Biseccion(k,u,n_pts,du,intervalos,tol=1e-10,max_iter=200)
    print("\nAlphas: ", alphas_biseccion)

    # Máscara para el plot
    mask = alphas >= 0.2
    alphas = alphas[mask]
    psi = psi[mask]

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, psi, lw=1.5)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\psi$')
    plt.title('Lista psi calculada con Numerov')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return alphas_biseccion

if __name__ == '__main__':
    
    V0 = 244
    a = 1e-10 
    h_b = 6.582e-16
    c = 299792458
    me = 0.511e6/(c**2)

    k = (2*me*(a**2)*V0)/(h_b**2)
    print("k = ",k)

    n_pts = 4001
    ulim = 2 # Punto más alejado donde se calucla el valor de la función de onda.
    u = np.linspace(-ulim,ulim,n_pts)
    du = (ulim*2)/(n_pts - 1)

    # Se han extraido los valores para otras iteraciones para no tener que calularlos cada vez

    alphas_biseccion = [np.float64(0.0981251656656685), np.float64(0.383098182184986), np.float64(0.8085638601215202)]
    #alphas_biseccion = calculos()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, alpha in zip(axes, alphas_biseccion):
        psi_pos = construirPsi(k, u, alpha, du, n_pts)
        ax.plot(u, psi_pos, linewidth=1.5)
        ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('u')
        ax.set_title(f'α = {alpha:.6f}')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('ψ(u)')
    plt.tight_layout()
    plt.show()