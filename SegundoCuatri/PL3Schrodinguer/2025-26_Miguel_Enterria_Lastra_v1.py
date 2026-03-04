'''
    Miguel Enterría Lastra

    Representación de función de onda e interacción con un potencial cuadrático.

    Construcción de la función de onda a partir de una función gaussiana o como polinomio de Hermite
'''



import numpy as np
import matplotlib.pyplot as plt

def norm(re,im,dx):
    '''
        Calcula la norma de la función de onda
    '''
    n = np.sqrt(np.sum(re**2 + im**2)*dx)
    return n

def normalizar(re,im,dx):
    '''
        Noramaliza la función
    '''
    n = norm(re,im,dx)
    return re/n, im/n

def modulo(re,im):
    '''
        Obtiene el módulo de la función para cada punto del espacio
    '''
    return np.sqrt(re**2 + im**2)

def d2(phi,dx):
    '''
        Derivada segunda en x de la función de onda
    '''
    der = np.zeros(len(phi))
    der[1:-1] = (phi[2:] + phi[:-2] - 2*phi[1:-1])
    der[0] = phi[1] + phi[-1] -2*phi[0]
    der[-1] = phi[-2] + phi[0] -2*phi[-1]
    return der/(dx**2)

def derx(phi,dx):
    '''
        Derivada de x de la función de onda
    '''
    der = np.zeros(len(phi))
    der[1:-1] = phi[2:] -phi[:-2]
    der[0] = phi[1] - phi[-1]
    der[-1] = phi[0] -phi[-2]
    return der /(2*dx)

def esp(re,im,f,dx):
    '''
        Calcula el valor esperado de X
    '''
    return np.sum((re**2 + im**2)*f)*dx

def esp_p(re,im,dx):
    '''
        Calcula el valor esperado de P
    '''
    dre = derx(re,dx)
    dim = derx(im,dx)
    return np.sum(re*dim - im*dre)*dx

def esp_p2(re,im,dx):
    '''
        Calcula el valor esperado de P²
    '''
    return -np.sum(re*d2(re,dx)+im*d2(im,dx))*dx

if __name__ == '__main__':

    # Datos del espacio #

    x_min = -5
    x_max = 5
    n_ptos = 1001
    mallado = np.linspace(x_min,x_max,n_ptos)
    dx = (x_max -x_min)/(n_ptos-1)
    dt = (dx**2)/2

    # Simulación #

    n_pasos = 50000
    n_p_rep = 50
    t_pausa = 0.01

    # Onda inicial #

    x0 = 0
    sigma = 0.5
    k0 = 10

    omega = 4
    raizOmega = np.sqrt(omega)

    V = 0.5*omega**2*mallado**2
    Vr = V/max(V)    

    Hermite = False # Variable para generar un polinomio de Hermite o no

    if Hermite:

        # -------- Hermite -------- #

        n = 5
        Herm0 = np.ones(n_ptos)
        Herm1 = 2*mallado*raizOmega
        Herm = Herm1
        Hermprev = Herm0
        if n == 0:
            Herm = Herm0
        if n >= 2:
            for i in range(2,n+1):
                HermTemp = Herm
                Herm = 2*mallado*raizOmega*Herm - 2*(i-1)*Hermprev
                Hermprev = HermTemp

        # ------------------------- #

        re = np.exp((-omega*mallado**2)/2)*Herm
        im = np.zeros(n_ptos)

    else:

        re = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.cos(k0*mallado)
        im = np.exp(-0.5*((mallado - x0)/sigma)**2)*np.sin(k0*mallado)


    print(f"Norma antes de normalizar: {norm(re,im,dx)}")
    re, im = normalizar(re,im,dx)
    print(f"Norma después de normalizar: {norm(re,im,dx)}")

    norma = []
    tiempo = []
    espx = []
    espp = []
    desx = []
    desp = []
    incert = []

    # Precalcular factores constantes
    dt_medio = dt/2
    dt_cuarto = dt/4
    dt_V = dt*V
    dt_V_medio = dt_V/2

    fig = plt.figure(figsize=(10,16), clear=True)
    ax = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,3,4)
    ax3 = plt.subplot(2,3,5)
    ax4 = plt.subplot(2,3,6)

    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    ax.set_xlim(mallado[0], mallado[-1])
    ax2.set_xlim(0,dt)
    ax3.set_xlim(0,dt)
    ax4.set_xlim(0,dt)

    norm_plot=ax2.plot([], [], c='purple', label='Norma')
    espx_plot = ax3.plot([], [], c = 'blue', label='<x>')
    espp_plot = ax3.plot([], [], c = 'red', label='<p>')
    desx_plot = ax4.plot([],[], c = 'blue', label='Δx')
    desp_plot = ax4.plot([], [], c = 'red', label='Δp')
    incert_plot = ax4.plot([], [], c = 'black', label='Δx*Δp')

    ax.plot(mallado, Vr, c='teal',label=r'$V(x)=\frac{1}{2}\omega^2x^2$', ls='dashed')

    ax.set_xlabel("Posición")
    ax.set_ylabel("Phi")
    ax.set_title("Onda")
    ax.legend()

    ax2.set_xlabel("Tiempo")
    ax2.set_ylabel("Norma")
    ax2.set_title("Evolución de la Norma")
    #ax2.set_ylim(0.999, 1.001)
    ax2.legend()

    ax3.set_xlabel("Tiempo")
    ax3.set_ylabel("Valor esperado")
    ax3.set_title("Evolución del valor esperado")
    ax3.legend()

    ax4.set_xlabel("Tiempo")
    ax4.set_ylabel("Desviación estandar")
    ax4.set_title("Evolución de la desviación estandar")
    ax4.legend()

    mod = modulo(re,im)

    OndaRe, = ax.plot(mallado, re, color="blue", label = 'Re(φ)')    
    OndaIm, = ax.plot(mallado, im, color="red", label = 'Im(φ)') 
    Onda, = ax.plot(mallado, mod, color="black", label = '|φ|')       

    t = 0

    plt.ion()

    for i in range(n_pasos):
        t += dt

        im = im + dt_cuarto*d2(re,dx) - dt_V_medio*re
        re = re - dt_medio*d2(im,dx) + dt_V*im
        im = im + dt_cuarto*d2(re,dx) - dt_V_medio*re

        # Solo calcular cuando sea necesario mostrar
        if i%n_p_rep == 0:
            mod = modulo(re,im)
            norma.append(norm(re,im,dx))
            tiempo.append(t)
            espx.append(esp(re,im,mallado,dx))
            espp.append(esp_p(re,im,dx))
            dex = np.sqrt(np.maximum((esp(re,im,mallado**2,dx) - esp(re,im,mallado,dx)**2),0.0))
            dep = np.sqrt(np.maximum((esp_p2(re,im,dx) - esp_p(re,im,dx)**2),0.0))
            desx.append(dex) #<x²> - <x>²
            desp.append(dep) #<p²> - <p>²
            incert.append(dex*dep)

        if i%n_p_rep == 0:
            # Plot Onda y potencial
            OndaRe.set_data(mallado, re)
            OndaIm.set_data(mallado, im)
            Onda.set_data(mallado, mod)

            # Plot Norma
            norm_plot[0].set_data(tiempo, norma)
            ax2.set_xlim(0,tiempo[-1])
            ax2.set_ylim(min(norma),max(norma))

            # Plot valores esperados
            espx_plot[0].set_data(tiempo,espx)
            espp_plot[0].set_data(tiempo,espp)
            minimo = min(min(espp),min(espx))
            maximo = max(max(espx),max(espp))
            ax3.set_ylim(minimo-0.1*minimo,maximo+0.1*maximo)
            ax3.set_xlim(0,tiempo[-1])

            # Plot incertidumbre
            desx_plot[0].set_data(tiempo,desx)
            desp_plot[0].set_data(tiempo,desp)
            incert_plot[0].set_data(tiempo,incert)
            minimo = min(min(desx),min(desp))
            maximo = max(max(desx),max(desp))
            ax4.set_ylim(minimo-0.1*minimo,maximo+0.1*maximo)
            ax4.set_xlim(0,tiempo[-1])

            plt.pause(t_pausa)

    plt.ioff()        
