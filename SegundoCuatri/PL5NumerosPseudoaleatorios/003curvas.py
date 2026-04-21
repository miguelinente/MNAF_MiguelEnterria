import numpy as np
import matplotlib.pyplot as plt

def simulacion_cadena():
    N0 = 10000  
    tau_Bi = 7.5
    tau_Po = 190
    
    u_Bi = np.random.uniform(0, 1, N0)
    tiempos_Bi = -tau_Bi * np.log(u_Bi)
    
    u_Po = np.random.uniform(0, 1, N0)
    tiempo_Po_vivo = -tau_Po * np.log(u_Po) # El tiempo que el que vivirá cada núcleo de plonio creado
    
    tiempos_Po = tiempos_Bi + tiempo_Po_vivo # Tiempo total que durará cada uno de los núcleos vivo
    
    tiempos_sim = np.linspace(0, 50, 100)
    N_Bi = []
    N_Po = []
    
    for t in tiempos_sim:
        restantes_Bi = np.sum(tiempos_Bi > t)
        
        vivos_Po = np.sum((tiempos_Bi <= t) & (tiempos_Po > t)) # Los que se hayan desintegrado del bismuto y los que todavía no se han desintegrado
        
        N_Bi.append(restantes_Bi)
        N_Po.append(vivos_Po)

    # Cálculos teóricos
    omega_Bi = 1 / tau_Bi
    omega_Po = 1 / tau_Po
    teoria_Bi = N0 * np.exp(-omega_Bi * tiempos_sim)
    teoria_Po = (omega_Bi / (omega_Po - omega_Bi)) * N0 * (np.exp(-omega_Bi * tiempos_sim) - np.exp(-omega_Po * tiempos_sim))

    plt.figure(figsize=(10,6))
    plt.plot(tiempos_sim, N_Bi, label="Bismuto-210 (Simulado)")
    plt.plot(tiempos_sim, N_Po, label="Polonio-210 (Simulado)")
    plt.plot(tiempos_sim, teoria_Bi, 'k--', label="Bismuto-210 (Teoría)")
    plt.plot(tiempos_sim, teoria_Po, 'r--', label="Polonio-210 (Teoría)")
    
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Número de núcleos")
    plt.title("Simulación de Cadena Radiactiva Bi -> Po -> Pb")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulacion_cadena()