import numpy as np
import matplotlib.pyplot as plt

def C(k,u,alpha):
    return np.where((u > -0.5) & (u < 0.5), -k*alpha, k*(1-alpha))
    
def psi_umax(k,u,alpha,n,du,par = True):
    '''
        Cálculo del último valor de la función de onda en u_max

        Entrada:
            k,u,alpha,n,du,par : valores del sistema para crear la función de onda

        Salida: 
            psi[-1]: último valor de la función de onda
    '''
    psi = np.zeros(n)
    dpsi = np.zeros(n)
    if par:
        psi[0] = 1
        dpsi[0] = 0
    else:
        psi[0] = 0
        dpsi[0] = 1
    
    C_vals = C(k,u,alpha)
    for i in range(1,n):
        psi[i] = psi[i-1] + dpsi[i-1]*du
        dpsi[i] = dpsi[i-1] + C_vals[i-1]*psi[i-1]*du
    return psi[-1]

def psi_plot(k,u,alpha,n,du,par=True):
    '''
        Construcción de la parte positiva de la onda para plotear

        Entrada:
            k,u,alpha,n,du,par : valores del sistema para crear la función de onda

        Salida: 
            psi: parte positiva de la onda
    '''
    psi = np.zeros(n)
    dpsi = np.zeros(n)
    if par:
        psi[0] = 1
        dpsi[0] = 0
    else:
        psi[0] = 0
        dpsi[0] = 1
    
    C_vals = C(k,u,alpha)
    for i in range(1,n):
        psi[i] = psi[i-1] + dpsi[i-1]*du
        dpsi[i] = dpsi[i-1] + C_vals[i-1]*psi[i-1]*du
    return psi


def onda_completa(u, psi_pos, par=True):
    '''
        Genera la otra parte de la onda según si es par o impar

        Entradas:
            u = espacio
            psi_pos = mitad positiva de la función de onda
            par = paridad
    '''
    u_total = np.concatenate((-u[:0:-1], u))
    if par:
        psi_total = np.concatenate((psi_pos[:0:-1], psi_pos))
    else:
        psi_total = np.concatenate((-psi_pos[:0:-1], psi_pos))
    
    maximo = np.max(np.abs(psi_total))
    if maximo > 0:
        psi_total = psi_total / maximo
    
    return u_total, psi_total

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


def Biseccion(k,u,n,du,intervalo,par=True,tol=1e-10,max_iter=200):
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
        fL = psi_umax(k, u, alpha_L, n, du, par=par)
        fR = psi_umax(k, u, alpha_R, n, du, par=par)

        if fL * fR > 0:
            raise ValueError("No hay cambio de signo en este intervalo, introducir un intervalo correcto.")

        iter = 0
        while alpha_R - alpha_L > tol and iter < max_iter:
            alpha_M = alpha_L + (alpha_R - alpha_L)/2
            fM = psi_umax(k, u, alpha_M, n, du, par=par)

            if fL * fM > 0:
                alpha_L = alpha_M
                fL = fM
            else:
                alpha_R = alpha_M
                fR = fM

            iter += 1

        alphas.append((alpha_L + alpha_R)/2)
    
    return alphas


if __name__ == '__main__':

    # Valores iniciales #

    V0 = 244
    a = 1e-10 
    h_b = 6.582e-16
    c = 299792458
    me = 0.511e6/(c**2)

    k = (2*me*(a**2)*V0)/(h_b**2)
    print("k = ",k)

    n_pts = 10001
    umax = 2 # Punto más alejado donde se calucla el valor de la función de onda.
    u = np.linspace(0,umax,n_pts)
    du = umax/(n_pts - 1)

    n_alphas = 2000
    alphas = np.linspace(0.001, 0.999, n_alphas)

    psi_par = np.zeros(n_alphas)
    for i in range(n_alphas):
        psi_par[i] = psi_umax(k,u,alphas[i],n_pts,du)

    psi_impar = np.zeros(n_alphas)
    for i in range(n_alphas):
        psi_impar[i] = psi_umax(k,u,alphas[i],n_pts,du,par=False)

    # Bisección #

    intervalo_par = barridoInicial(psi_par,alphas)
    intervalo_impar = barridoInicial(psi_impar,alphas)

    print("\nIntervalos pares: ", intervalo_par)
    print("\nIntervalos impares: ", intervalo_impar)

    alphas_par_biseccion = Biseccion(k,u,n_pts,du,intervalo_par,par=True,tol=1e-10,max_iter=200)
    alphas_impar_biseccion = Biseccion(k,u,n_pts,du,intervalo_impar,par=False,tol=1e-10,max_iter=200)

    print("\nAlphas pares: ", alphas_par_biseccion)
    print("\nAlphas impares: ", alphas_impar_biseccion)

    plt.figure(figsize=(10,6))
    for alpha in alphas_par_biseccion:
        psi_pos = psi_plot(k, u, alpha, n_pts, du, par=True)
        u_total, psi_total = onda_completa(u, psi_pos, par=True)
        plt.plot(u_total, psi_total, label=f'α = {alpha:.6f}', linewidth=1.5)
    for alpha in alphas_impar_biseccion:
        psi_pos = psi_plot(k, u, alpha, n_pts, du, par=False)
        u_total, psi_total = onda_completa(u, psi_pos, par=False)
        plt.plot(u_total, psi_total, label=f'α = {alpha:.6f}', linewidth=1.5)

    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('u')
    plt.ylabel('ψ(u)')
    plt.title('Funciones de onda para los α obtenidos por bisección')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


   


    

