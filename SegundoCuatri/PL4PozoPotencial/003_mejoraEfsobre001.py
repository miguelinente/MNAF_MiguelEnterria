import numpy as np
import matplotlib.pyplot as plt

def C(k,u,alpha):
    return np.where((u > -0.5) & (u < 0.5), -k*alpha, k*(1-alpha))
    
def psi_umax(k,u,alpha,n,du,par = True):
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


if __name__ == '__main__':

    # Valores iniciales #

    V0 = 244
    a = 1e-10 
    h_b = 6.582e-16
    c = 299792458
    me = 0.511e6/(c**2)

    k = (2*me*(a**2)*V0)/(h_b**2)
    print("k = ",k)

    n_pts = 4001
    umax = 1 # El doble de la extensión del pozo
    u = np.linspace(0,umax,n_pts)
    du = umax/n_pts

    n_alphas = 2000
    alphas = np.linspace(0.001, 0.999, n_alphas)

    psi_par = np.zeros(n_alphas)
    for i in range(n_alphas):
        psi_par[i] = psi_umax(k,u,alphas[i],n_pts,du)

    psi_impar = np.zeros(n_alphas)
    for i in range(n_alphas):
        psi_impar[i] = psi_umax(k,u,alphas[i],n_pts,du,par=False)
    
    # Plot de ambas listas
    plt.figure(figsize=(10,6))
    plt.plot(alphas, psi_par, label='ψ par', linewidth=1.5)
    plt.plot(alphas, psi_impar, label='ψ impar', linewidth=1.5)
    plt.xlabel('α')
    plt.ylabel('ψ(u_max)')
    plt.title('Función de onda en u_max para estados pares e impares')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Bisección #

    #E = np.array([,0])

