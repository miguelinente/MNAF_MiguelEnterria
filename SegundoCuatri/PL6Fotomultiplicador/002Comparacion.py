import numpy as np
import matplotlib.pyplot as plt

def simular_pmt(v_list, n_inicial=1):
    """Simula un evento con una lista de ganancias por dínodo."""
    n_electrones = n_inicial
    for v in v_list:
        if n_electrones == 0:
            return 0
        # Simulación de Poisson vectorizada para rapidez
        n_electrones = np.random.poisson(v, n_electrones).sum()
    return n_electrones

def ejecutar_estudio_pmt(iteraciones=10000):
    # Configuración de los experimentos
    configs = {
        "Base (6x v=5)": [5, 5, 5, 5, 5, 5],
        "v=7 en D1":     [7, 5, 5, 5, 5, 5],
    }
    
    # También probamos v=7 en todas las posiciones para responder a tu duda
    for i in range(6):
        v_custom = [5]*6
        v_custom[i] = 7
        # Esto permite ver la tendencia (D1, D2, D3...)
    
    resultados_finales = {}
    umbral_realista = 25000

    print(f"{'Configuración':<20} | {'Int. Media':<12} | {'Ef. Ideal':<10} | {'Ef. Realista'}")
    print("-" * 65)

    plt.figure(figsize=(10, 6))

    for nombre, v_config in configs.items():
        res = np.array([simular_pmt(v_config) for _ in range(iteraciones)])
        
        int_media = np.mean(res)
        ef_ideal = np.mean(res >= 1) * 100
        ef_realista = np.mean(res >= umbral_realista) * 100
        
        print(f"{nombre:<20} | {int_media:>12.1f} | {ef_ideal:>9.2f}% | {ef_realista:>11.2f}%")
        
        # Guardamos para el histograma de la configuración base
        if nombre == "Base (6x v=5)":
            plt.hist(res, bins=50, alpha=0.7, label=nombre, color='blue', edgecolor='black')

    plt.axvline(umbral_realista, color='red', linestyle='--', label=f'Umbral Realista ({umbral_realista})')
    plt.title("Distribución de Electrones en el Ánodo (Configuración Base)")
    plt.xlabel("Número de Electrones")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ejecutar_estudio_pmt()