import numpy as np
import matplotlib.pyplot as plt

#datos, incertidumbre fija (usar la obtenida)
q = np.array([2.1,2.4,4.9,6.6,1.8,1.9,2.2,2.,4.,5]) #valores de q que hemos calculado q_i unidades de carga (10^-19C) 
deltaq = np.ones(np.size(q))*0.1 #incertidumbres

#Función a minimizar
def F(e): #e es la carga fundamental
	sum=0.
	for i in range(np.size(q)):
		sum+= (q[i]- e*np.rint(q[i]/e))**2
		return sum

#barrido entre la menor y 2.5 unidades
e = np.linspace(np.amin(q-deltaq),2.5,1000)     #1000 valores ~precision

#Gráfica F(e) vs e (carga fundamental)
plt.plot(e,F(e))
plt.show()

#minimización y obtención de n_i
min = np.where(F(e)==np.amin(F(e)))
mye = e[min[0]] #e para F mínimo  Primera estimación
print(u"El mínimo de la función es:",mye)

#estimación de la carga fundamental de cada gota y su incertidumbre
n = np.rint(q/mye) #n_i enteros de e
e= q/n #estimaciones de e_i
deltae = deltaq/n #incertidumbre en e_i

#Media ponderada de los e_i
w = 1/deltae/deltae #peso ~ 1/incert2
num = np.sum(w*e) #numerador
den = np.sum(w) #denominador
ave = num/den #valor final de e
incert = np.sqrt(den)

print (ave,"+-",1/incert)