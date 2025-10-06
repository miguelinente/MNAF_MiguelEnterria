import numpy as np
import sympy as sp
import numpy.polynomial as P


'''
x = sp.Symbol('x')

a = 2

p = sp.poly(1+2* x +3* x **2)
q = sp.Poly([2,5],x)
p_2 = sp.Poly([3,2,1],x)
p_3 = sp.poly(np.prod(x-np.array([1,2])))

print(p(a))

print(f"El grado es: {p.degree()}")
print(f"El intervalo de raices es: {p.intervals()}")
print(f"Polinimio shifted: {p.shift(a)}")
print(f"Expresión de horner: {sp.horner(p)}")
d,r = sp.div(p,q)
print(f"Dividendo{d}")
print(f"Resto{r}")

print(f"Derivada{p.diff()}")
print(f"Integral{p.integrate()}")

y0 = 4
y1= 2
y2 = 3
print(sp.interpolate([y0, y1, y2],x))

b= 0
f = sp.sin(x)
n = 9
ser = sp.series(f,x,b,n)
print(f"Polinomio de taylor del seno de x: {ser}")
#pol = sp.remove0(ser)
#print(f"polinomio con 0 eliminado: {pol}")


p = P.Polynomial([1,2,3])
print(f"Polinomio: {p}")
p = P.Polynomial.fromroots([1,2])
print(f"Polinomio: {p}")
print(f"Polinomio en punto x = a: {p(a)}")
print(p.coef)
print(p.domain)
print(p.window)

n= 1
a = 0
c = 0
p_int = p.integ(m=n, k=[c]*n)
print(p_int)
print(p.deriv(1))
print(p.basis(2))

'''

