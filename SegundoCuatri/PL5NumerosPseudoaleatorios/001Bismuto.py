import numpy as np
import matplotlib.pyplot as plt

def desintegracionBi(plot = True):

    N0 = 100
    t = np.zeros(N0)
    tau = 7.5 #*24*60*60 # En segundos
    print(f"\nvida media (τ) = {tau}\n")
    u = np.random.uniform(0,1,N0)
    tiempos_desintegracion = -tau * np.log(u)

    tiempos_simulacion = np.linspace(0,40,40) #0 a 40 días
    nucleos_restantes = []

    for t_sim in tiempos_simulacion:
        restantes = np.sum(tiempos_desintegracion > t_sim)
        nucleos_restantes.append(restantes)
   
    teoria = N0 * np.exp(-tiempos_simulacion / tau) # Curva teórica
    if plot == True:
        plt.step(tiempos_simulacion, nucleos_restantes, label="Simulación (Monte Carlo)")
        plt.plot(tiempos_simulacion, teoria, 'r--', label="Teoría ($N_0 e^{-t/\\tau}$)")
        plt.xlabel("Tiempo (días)")
        plt.ylabel("Número de núcleos (N)")
        plt.legend()
        plt.show()
    

    return nucleos_restantes, tiempos_simulacion



if __name__ == "__main__":
    
    desintegracionBi()
