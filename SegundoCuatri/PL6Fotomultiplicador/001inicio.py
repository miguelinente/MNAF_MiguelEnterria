import numpy as np

def poisson_manual(v):
    """
    Genera un número aleatorio Poisson usando el algoritmo de teoría.
    Basado en el producto de uniformes hasta que A < e^-v.
    """
    k = 1
    A = 1
    lim = np.exp(-v)
    while True:
        u = np.random.uniform(0, 1)
        A *= u
        if A < lim:
            return k - 1
        k += 1

def simular_evento(v=5, n_dinodos=6, n_inicial=1, metodo='numpy'):
    """
    Simula la cascada de electrones en el fotomultiplicador.
    metodo: 'numpy' para velocidad, 'manual' para el algoritmo de teoría.
    """
    n_electrones = n_inicial
    for _ in range(n_dinodos):
        if n_electrones == 0:
            return 0
        
        if metodo == 'numpy':
            n_electrones = np.random.poisson(v, n_electrones).sum()
        else:
            secundarios = 0
            for _ in range(n_electrones):
                secundarios += poisson_manual(v)
            n_electrones = secundarios
            
    return n_electrones

def calcular_estadisticas(iteraciones=10000, v=5, umbral_realista=25000, metodo='numpy'):
    resultados = []
    print(f"Simulando {iteraciones} eventos usando método: {metodo}...")
    
    for _ in range(iteraciones):
        resultados.append(simular_evento(v=v, metodo=metodo))
    
    resultados = np.array(resultados)
    
    eficiencia_ideal = np.mean(resultados >= 1)
    eficiencia_realista = np.mean(resultados >= umbral_realista)
    intensidad_media = np.mean(resultados)
    
    print(f"--- Resultados con v={v} ({metodo}) ---")
    print(f"Intensidad media final: {intensidad_media:.2f} electrones")
    print(f"Eficiencia Ideal (n >= 1): {eficiencia_ideal * 100:.2f}%")
    print(f"Eficiencia Realista (n >= {umbral_realista}): {eficiencia_realista * 100:.2f}%")
    
    return resultados

if __name__ == '__main__':
    res = calcular_estadisticas(iteraciones=5000, v=5, metodo='numpy')