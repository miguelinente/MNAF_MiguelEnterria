import numpy as np
import math as m

def CalcE(e, h_bar, me, a):
    E = (e**2)*(2*h_bar**2)/()

if __name__ == "__main__":

    a = 1e-10 # metros
    h_bar = 1.05457e-34
    V0 = 244*1.60218e-19 #Julios
    me = 9.109e-31
    alpha = 0.0979

    E1 = 1.252
    E2 = 3.595
    E3 = 2.475

    k = m.sqrt((2*me*V0)/(h_bar**2))*m.sqrt(alpha)
    print("K = ", k)

    c = m.sqrt((2*me*V0)/(h_bar**2))*m.sqrt(1 - alpha)
    value = k*m.tan(k*a/2)

    print("Valorvalue")