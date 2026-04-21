import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # Librería fundamental para ajustes

def desintegracionBi_con_ajuste():
    N0_real = 1000
    tau_real = 7.5 
    u = np.random.uniform(0, 1, N0_real)
    tiempos_desintegracion = -tau_real * np.log(u)

    tiempos_simulacion = np.linspace(0, 40, 2000)
    nucleos_restantes = np.array([np.sum(tiempos_desintegracion > t_sim) for t_sim in tiempos_simulacion])

    # Definimos la función teórica que queremos encontrar
    def modelo_exponencial(t, N0, tau):
        return N0 * np.exp(-t / tau)

    # popt: parámetros óptimos encontrados [N0, tau]
    # pcov: matriz de covarianza (para calcular el error del ajuste)
    popt, pcov = curve_fit(modelo_exponencial, tiempos_simulacion, nucleos_restantes, p0=[1000, 7])
    
    N0_fit, tau_fit = popt
    error_tau = np.sqrt(pcov[1,1]) # Desviación estándar de tau

    # --- Visualización ---
    plt.step(tiempos_simulacion, nucleos_restantes, label="Simulación (Monte Carlo)", color='gray', alpha=0.6)
    plt.plot(tiempos_simulacion, modelo_exponencial(tiempos_simulacion, *popt), 'g-', 
             label=f"Ajuste (τ={tau_fit:.2f} ± {error_tau:.2f} días)", linewidth=2)
    plt.plot(tiempos_simulacion, N0_real * np.exp(-tiempos_simulacion/tau_real), 'r--', label="Teoría Real")
    
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Número de núcleos (N)")
    plt.title("Ajuste de la constante de desintegración")
    plt.legend()
    plt.show()

    print(f"Vida media obtenida del ajuste: {tau_fit:.3f} días")

if __name__ == "__main__":
    desintegracionBi_con_ajuste()