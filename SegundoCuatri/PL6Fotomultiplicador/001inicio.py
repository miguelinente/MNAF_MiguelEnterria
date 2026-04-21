import numpy as np

def poisson(v):

    k = 1
    A = 1
    lim = np.exp(-v)

    while True:
        u = np.random.uniform(0,1)
        A *= u
        if A < lim:
            return k - 1
        k += 1


def fotomultiplicador(v = 5, n_dinodos = 6):
    
    n_electrones = 25000

    print(f'Estado inicial {n_electrones} electrones')

    for i in range(n_dinodos):
        secundarios = 0

        for _ in range(n_electrones):
            secundarios += poisson(v)
        
        n_electrones = secundarios
        print(f"Dínodo {i+1}: {n_electrones} electrones")

        if n_electrones == 0:
            break

    return n_electrones

if __name__ == '__main__':
    resultado = fotomultiplicador(v=5, n_dinodos=6)
    print(f"\nSeñal final en el ánodo: {resultado} electrones")