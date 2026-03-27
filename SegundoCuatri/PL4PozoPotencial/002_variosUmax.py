import numpy as np
import matplotlib.pyplot as plt

def C(k,u,alpha):
    if u > -0.5 and u < 0.5:
        return -k*alpha
    else:
        return k*(1-alpha)
    
def psi_umax(k,u,alpha,n,du,par = True):
    psi = []
    dpsi = []
    if par:
        psi.append(1)
        dpsi.append(0)
        for i in range(1,n):
            psi.append(psi[i-1] + dpsi[i-1]*du) 
            dpsi.append(dpsi[i-1] + C(k,u[i-1],alpha)*psi[i-1]*du)
        return psi[-1]
    else:
        psi.append(0)
        dpsi.append(1)
        for i in range(1,n):
            psi.append(psi[i-1] + dpsi[i-1]*du) 
            dpsi.append(dpsi[i-1] + C(k,u[i-1],alpha)*psi[i-1]*du) 
        return psi[-1]


if __name__ == '__main__':

    # Valores iniciales #

    V0 = 244 # eV
    a = 10**(-10) 
    h_b = 6.582*(10**(-16)) #eVs
    me = 0.511*(10**6)/(299792458**2) # MeV/c²

    k = (2*me*(a**2)*V0)/(h_b**2)
    print("k    = ", k)

    n_pts = 4001
    n_alphas = 2000
    alphas = np.linspace(0.001, 0.999, n_alphas)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for idx, umax in enumerate([1, 2, 3, 4]):
        u = np.linspace(0,umax,n_pts)
        du = umax/n_pts

        psi_par = []
        for i in range(n_alphas):
            psi_par.append(psi_umax(k,u,alphas[i],n_pts,du))

        psi_impar = []
        for i in range(n_alphas):
            psi_impar.append(psi_umax(k,u,alphas[i],n_pts,du,par=False))
        
        axs[idx].plot(alphas, psi_par, label='ψ par', linewidth=1.5)
        axs[idx].plot(alphas, psi_impar, label='ψ impar', linewidth=1.5)
        axs[idx].set_xlabel('α')
        axs[idx].set_ylabel('ψ(u_max)')
        axs[idx].set_title(f'umax = {umax}')
        axs[idx].legend()
        axs[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

