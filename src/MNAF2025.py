# -*- coding: utf-8 -*-
''' Archivo que contiene todas la funciones requeridas por los ejercicios '''

from __init__ import _hello
import math
import numpy as np
import numpy.polynomial as P
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from scipy.interpolate import lagrange, CubicSpline
import scipy.integrate as scin
import scipy.linalg as spla
from numpy.polynomial.legendre import leggauss

def base_lagrange(soporte, plot=False):

    '''
    Función que construye la base de polinomios de Lagrange dado un soporte.

    Parámetros:
        Entrada:
            Soporte: puntos del soporte sobre los que se calcula la base.
        Salida: 
            B: Lista con las expresiones simbólicas de los polinomios resultantes.
    '''
    # Transforamción a lista y copia de soporte a x_list
    match soporte:
        case list():
            print("Era una lista")
            x_list = soporte
        case np.ndarray():
            print("Era un array")
            x_list = soporte.tolist()
        case tuple():
            print("Era una tupla")
            x_list = list(soporte)
        case _:
            raise ValueError("Tipo de dato inválido")
        
    if len(set(x_list)) != len(x_list):
        raise ValueError("El soporte no puede tener puntos repetidos")
    
    print("Lista resultante:", x_list)

    xsym = sp.Symbol('x')
    B = []
    n = len(x_list)

    # Cálculo de la base lagrange
    for i in range(n):
        xi = x_list[i]
        Li = 1
        for xj in x_list:
            if xj != xi: 
                Li *= ((xsym-xj)/(xi-xj))
        B.append(sp.simplify(Li))

    fig, ax = plt.subplots(2, 2, num=2, figsize=(16, 10), clear=True)
    axes = ax.flatten()
    x = np.linspace(x_list[0], x_list[-1], 201)
    if plot: 
    # Representación gráfica
        for i, Li in enumerate(B):
            Li_num = sp.lambdify(xsym, Li, 'numpy')
            y = Li_num(x)

            axes[i].plot(x,y, label=f"L{i}(x)")
            axes[i].set_title(f"L{i}(x)")
            axes[i].grid()
            axes[i].legend()

        fig.tight_layout()
        plt.show()

    return B

def itp_Tchebisev(fun, npts, a, b):
    '''
    Cáculo del interpolante polinómico utilizando nodos de Tchebisev.

    Parámetros: 
        Entrada: 
            fun  : función numérica que se quiere interpolar.
            npts : número de puntos (nodos) de interpolación.
            a,b  : extremos del intervalo [a,b] donde se interpola.
        Salida:
            Gráfica del polinomio.
    '''
    # Se calcula el polinomio de chebisev del orden correspondiente al número de puntos que tengamos y luego se sacan sus raices
    pol = P.Chebyshev.basis(npts)
    x_cheb_array = pol.roots()
    # Se convierte a lista 
    x_cheb = x_cheb_array.tolist()
    # Se comprueba el intervalo para ver si hay que hacer transformación o no
    if a != -1 or b != 1:
        for i, xi in enumerate(x_cheb):
            x_cheb[i] = ((a+b)/2) + ((b-a)/2)*x_cheb[i]

    x_cheb = np.asarray(x_cheb, dtype = float)
    y_cheb = fun(x_cheb)
    itp_lag = lagrange(x_cheb,y_cheb)

    fig, ax = plt.subplots()

    x = np.linspace(x_cheb[0], x_cheb[-1], 201)
    # Representación gráfica
    y = itp_lag(x)
    ax.plot(x,y, label="Tchebisev")
    ax.set_title(f"Tchebisev")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()

def itp_parametrica(data, bc_type='natural', u =None):
    '''
    Calcula el interpolante paramétrico de unos datos mediante splines cúbicas.

    Parámetros:
        Entrada:
            data : array con forma (dim, N), donde cada fila es una coordenada
                y cada columna un punto de la curva.
            bc_type : tipo de condición de contorno para CubicSpline
                ('natural', 'clamped', 'periodic', etc.).
            u : parámetro asociado a los puntos. Si es None, se construye
                proporcional a la distancia entre puntos y escalado en [0,1].
    '''
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data debe ser un array bidimensional (dim, N)")

    dim, N = data.shape

    if N < 2:
        raise ValueError("Se necesitan al menos dos puntos para interpolar")

    # Construcción del parámetro si no se proporciona
    if u is None:
        dif = np.diff(data, axis=1)
        dist = np.linalg.norm(dif, axis=0)
        u = np.concatenate(([0.0], np.cumsum(dist)))

        if u[-1] > 0:
            u = u / u[-1]
    else:
        u = np.asarray(u, dtype=float)
        if u.size != N:
            raise ValueError("El tamaño de u debe coincidir con el número de puntos")

    splines = []
    for k in range(dim):
        cs = CubicSpline(u, data[k], bc_type=bc_type)
        splines.append(cs)

    # Función vectorial interpolante
    def funitp(t):
        t = np.asarray(t)
        return np.array([s(t) for s in splines])

    return funitp, u

def dncoef_base(soporte,puntos,orden):
    '''
    Calcula los coeficientes de una regla de derivación numérica de orden "orden" en uno o varios "puntos",
    usando la base de Lagrange asociada a "soporte". Devuelve una lista con coeficientes.

    Entrada:
        soporte : lista/tupla/np.ndarray con los nodos {x0,...,xn}.
        puntos  : un número (punto único) o un iterable de puntos donde se quiere la derivada.
        orden   : entero >=0 y < número de nodos del soporte.
    Salida:
        coef    : si puntos es escalar -> lista de longitud n con coeficientes.
                  si puntos es iterable -> lista de listas, una por cada punto.
    '''
    if not isinstance(soporte, (list, tuple, np.ndarray)):
        raise ValueError("soporte debe ser list, tuple o np.ndarray")

    if not isinstance(orden, int):
        raise ValueError("orden debe ser un entero")

    if isinstance(soporte, np.ndarray):
        x_list = soporte.tolist()
    elif isinstance(soporte, tuple):
        x_list = list(soporte)
    else:
        x_list = soporte

    n = len(x_list)
    if n < 2:
        raise ValueError("El soporte debe tener al menos 2 puntos")
    if orden < 0 or orden >= n:
        raise ValueError("El orden debe ser >=0 y menor que el número de nodos del soporte")
    
    B = base_lagrange(soporte)

    xs = sp.Symbol('x')

    #Lista con las derivadas de los polinomios de Lagrange
    dB = [sp.diff(Li, xs, orden) for Li in B]

    if np.isscalar(puntos):
        # Un único punto: devolvemos lista de coeficientes
        p = float(puntos)
        coef = [float(di.subs(xs, p)) for di in dB]
        return coef
    else:
        # Varios puntos: devolvemos lista de listas
        pts = np.asarray(puntos, dtype=float)
        coef = []
        for p in pts:
            coef.append([float(di.subs(xs, float(p))) for di in dB])
        return coef

def deriva2(fun,puntos,h):
    '''
    Calcula la derivada segunda de una función en diferentes puntos según ciertas reglas numéricas.

    Entrada:
        fun : función numérica
        puntos : puntos donde calcular la derivada segunda
        h : paso de la aproximación
    Salida:
        Lista con los resultados para cada uno de los puntos
    '''
    puntos_scalar = np.isscalar(puntos)
    h_scalar = np.isscalar(h)
    
    if (not puntos_scalar) and (not h_scalar):
        raise ValueError("No se permite que ambos sean vectores simultaneamente")
    
    if puntos_scalar:
        x = np.asarray([puntos], dtype=float)
    else:
        x = np.asarray(puntos, dtype=float)

    if h_scalar:
        hh = float(h)
    else:
        x = np.full_like(hh, float(x[0]), dtype=float)
        hh = np.asarray(h, dtype=float)
    # --- Regla 1: 3 puntos adelantada ---
    r1 = (fun(x) - 2*fun(x + hh) + fun(x + 2*hh)) / (hh**2)

    # --- Regla 2: 3 puntos centrada ---
    r2 = (fun(x - hh) - 2*fun(x) + fun(x + hh)) / (hh**2)

    # --- Regla 3: 4 puntos adelantada ---
    r3 = (2*fun(x) - 5*fun(x + hh) + 4*fun(x + 2*hh) - fun(x + 3*hh)) / (hh**2)

    # --- Regla 4: 5 puntos centrada ---
    r4 = (-fun(x - 2*hh) + 16*fun(x - hh) - 30*fun(x) + 16*fun(x + hh) - fun(x + 2*hh)) / (12*(hh**2))

    return [r1.tolist(), r2.tolist(), r3.tolist(), r4.tolist()]

def incoef_base(soporte,puntos,a,b):
    '''
    Calcula los coeficientes de una regla de integración numérica para calcular la integral en un iontervalo [a,b] 
    utilizando los polinomios de Lagrange dado un soporte

    Entrada:
        soporte : lista/tupla/np.ndarray con los nodos
        a, b    : extremos del intervalo de integración
    Salida:
        coef    : lista de coeficientes calculados
    '''
    if not isinstance(soporte, (list, tuple, np.ndarray)):
        raise ValueError("soporte debe ser list, tuple o np.ndarray")

    B = base_lagrange(soporte)

    xs = sp.Symbol('x')
    coef = []

    for Lk in B:
        Ik = sp.integrate(Lk, (xs, a, b))
        coef.append(float(Ik))

    return coef

def in_romberg(fun, a, b, nivel=10, tol=1e-6):
    '''
    Calcula la integral definida de fun en [a,b] usando el método de Romberg.

    Entrada:
        fun   : función numérica
        a,b   : extremos de integración
        nivel : número máximo de niveles 
        tol   : tolerancia para el error 
    Salida:
        I     : valor estimado de la integral
        err   : estimación del error
        N     : tabla de Romberg (matriz)
    '''
    N = np.zeros((nivel,nivel), dtype=float)

    h = b-a
    N[0, 0]= 0.5*h*(fun(a) + fun(b))
    p=1

    for n in range(1, nivel):
        h *= 0.5
        suma = 0.0

        for i in range(1, p+1):
            suma += fun(a + (2*i - 1) * h)
        
        N[n,0] = 0.5 * N[n-1, 0] + h*suma
        q = 1

        for j in range(1, n+1):
            q *= 4
            N[n, j] = N[n, j-1] + (N[n, j-1] - N[n-1, j-1]) / (q - 1)
        
        err = abs(N[n, n] - N[n-1, n-1])
        if err < tol:
            return N[n, n], err, N[:n+1, :n+1]

        p *= 2

    return N[nivel-1, nivel-1], err, N

def paracaidista(y0,v0,m,cx,At,apertura=1500, rovar = False):
    '''
    Calcula la caída vertical de un paracaidista con rozamiento cuadrático,
    cambiando el coeficiente aerodinámico al abrir el paracaídas a una cierta altura.

    Entrada:
        y0: altura inicial (m)
        v0: velocidad inicial (m/s)
        m: masa (kg)
        cx: coeficiente de rozamiento
        At: área transversal
        apertura: altura a la que se abre el paracaidas
        rovar: define si se usa densidad variable del aire o no
    Salida:
        [v_max, v_impacto, t_apertura, t_total]
    '''
    cx = list(cx)
    if len(cx) != 2:
        raise ValueError("cx debe contener dos valores: (antes, despues) de la apertura.")
    if m <= 0 or At <= 0:
        raise ValueError("m y At deben ser positivos.")
    if y0 <= 0:
        return [abs(v0), abs(v0), 0.0, 0.0]

    grav = 9.81
    rho0 = 1.225
    gamma = 1.0 / 8243.0 

    def rho(y):
        if not rovar:
            return rho0
        y = max(float(y),0.0)
        return rho0*np.exp(-gamma*y)
    
    def sedo(t, Y, m_, cx_, At_):
        kw = 0.5*cx_*rho(Y[0])*At_
        dY = np.zeros(2, dtype=float)
        dY[0] = Y[1]
        dY[1] = -grav - (kw/m_) * Y[1] * abs(Y[1])
        return dY
    
    def abrePar(t, Y, m_, cx_, At_):
        return Y[0] - apertura
    abrePar.terminal = True
    abrePar.direction = -1

    def impactoSuelo(t, Y, m_, cx_, At_):
        return Y[0]
    impactoSuelo.terminal = True
    impactoSuelo.direction = -1

    Yini = np.array([float(y0), float(v0)], dtype=float)
    tfin= 10000.0

    sol1 = scin.solve_ivp(sedo, [0.0, tfin], Yini,args=(m, cx[0], At),events=abrePar)

    t_ap = float(sol1.t_events[0][0])
    Y_ap = sol1.y_events[0][0]

    sol2 = scin.solve_ivp(sedo, [t_ap, tfin], Y_ap,args=(m, cx[1], At),events=impactoSuelo,dense_output=True,method="RK45")

    t_total = float(sol2.t_events[0][0])
    v_impacto = abs(sol2.y_events[0][0][1])

    v_hist_total = np.hstack([sol1.y[1], sol2.y[1]])
    v_max = abs(np.min(v_hist_total))

    return [v_max, v_impacto, t_ap, t_total]

def enlsolver(funx, a, b, meth, maxiter, tol):
    '''
    Resuelve funciones por distintos métodos de intervalo.

    Entrada:
        funx: función a resolver
        a,b: extremos del intervalo
        meth: método de resolución
        maxiter: número máximo de iteraciones
        tol: tolerancias del intervalo

    Salida:
        r: cero aproximado
        info: motivo por el que finalizó la iteración
        suc: lista con la sucesión de valores
    '''
    Ex, ex, EF = tol
    suc = []

    if meth not in ("di", "rf", "fm"):
        return None, -2, []

    fa = funx(a)
    fb = funx(b)

    if abs(fa) < EF:
        return float(a), 1, [float(a)]
    if abs(fb) < EF:
        return float(b), 1, [float(b)]

    if fa * fb > 0:
        return None, -1, []

    fx_prev = None

    for _ in range(int(maxiter)):
        if meth == "di":
            xn = 0.5 * (a + b)
        else:
            xn = (a * fb - b * fa) / (fb - fa)

        fx = funx(xn)
        suc.append(float(xn))

        if abs(fx) < EF:
            return float(xn), 1, suc

        if fx * fa > 0:
            a, fa = xn, fx
            if meth == "fm" and (fx_prev is not None) and (fx * fx_prev > 0):
                fb *= 0.5
        else:
            b, fb = xn, fx
            if meth == "fm" and (fx_prev is not None) and (fx * fx_prev > 0):
                fa *= 0.5

        if meth == "fm":
            fx_prev = fx

        xtol = max(Ex, ex * abs(xn))
        if abs(b - a) < xtol:
            return float(xn), 0, suc
        
    return float(suc[-1]), 2, suc

def enlsteffensen(funx, x0, maxiter=128, tol=(1e-9, 1e-5, 1e-12)):
    """
    Método de Steffensen para resolver f(x)=0.

    Entrada:
        funx : función a resolver
        x0 : punto inicial.
        maxiter: máximo número de iteraciones (por defecto 128).
        tol: tolerancias

    Salida:
        r : cero aproximado.
        info: motivo por el que finalizó la iteración
        suc: sucesión {x_n}
    """
    Ex, ex, EF = tol
    x_prev = float(x0)
    suc = [x_prev]

    for _ in range(int(maxiter)):
        fx = funx(x_prev)

        if abs(fx) < EF:
            return x_prev, 0, suc  
        fy = funx(x_prev + fx)

        if fx == 0:
            return x_prev, 0, suc

        gx = fy / fx - 1.0

        if gx == 0:
            return suc[-1], 2, suc

        x_new = x_prev - fx / gx
        suc.append(float(x_new))

        dx = abs(x_new - x_prev)

        if dx < Ex:
            return float(x_new), 0, suc

        if dx < ex * abs(x_new):
            return float(x_new), 1, suc

        x_prev = x_new

    return float(suc[-1]), 2, suc

def sor_interval(A):
    """
    Calcula el intervalo de omega donde SOR converge y el omega óptimo.

    Entrada:
        A : matriz cuadrada

    Salida:
        inter : [wi, wf] intervalo de omegas convergentes
        ropt  : rho_min
        wopt  : omega que logra rho_min
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada.")
    if np.any(np.diag(A) == 0):
        return [], None, None

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)   
    U = -np.triu(A,  1)   

    W = np.round(np.arange(0.01, 1.99 + 1e-12, 0.01), 2)

    # 2) Inicialización
    ropt = np.inf
    wopt = None
    valid = []  

    for w in W:
        M = D - w * L
        N = (1.0 - w) * D + w * U

        try:
            Bw = np.linalg.solve(M, N)
        except np.linalg.LinAlgError:
            continue

        eigs = np.linalg.eigvals(Bw)
        rho = float(np.max(np.abs(eigs)))

        if rho < 1.0:
            valid.append(float(w))
            if rho < ropt:
                ropt = rho
                wopt = float(w)

    if len(valid) == 0:
        return [], None, None

    valid_sorted = np.array(sorted(valid))
    blocks = []
    start = valid_sorted[0]
    prev = valid_sorted[0]
    for x in valid_sorted[1:]:
        if abs(x - prev - 0.01) < 1e-9:
            prev = x
        else:
            blocks.append((start, prev))
            start = x
            prev = x
    blocks.append((start, prev))

    lengths = [int(round((b1 - b0) / 0.01)) for (b0, b1) in blocks]
    idx = int(np.argmax(lengths))
    wi, wf = blocks[idx]

    inter = [float(wi), float(wf)]
    if wopt is None or not np.isfinite(ropt):
        return inter, None, None

    return inter, float(ropt), float(np.round(wopt, 2))

def autoval_potencia(A, delta=np.inf, tol=(1e-6, 2), niter=100):
    """
    Aproxima el autovalor de A más cercano a delta mediante:
      - delta = ±inf  -> método de la potencia (mayor |lambda|)
      - delta finito  -> potencia inversa con desplazamiento (A - delta I)^(-1)

    Entrada:
        A : matriz de estudio.
        delta : referencia del autovalor: ±inf (mayor |.|), 0 (menor no nulo), otro valor (más cercano).
        tol : tolerancia del error
        niter : máximo de iteraciones.

    Salida:
        valor : autovalor aproximado.
        vector : autovector aproximado.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada.")

    if isinstance(tol, (int, float, np.floating)):
        eps = float(tol)
        p = 2
    else:
        eps = float(tol[0])
        p = tol[1]
        if p == np.inf:
            p = np.inf
        elif p not in (1, 2):
            raise ValueError("p debe ser 1, 2 o np.inf.")

    w0 = np.random.rand(n)
    u_prev = w0 / np.linalg.norm(w0, 2)

    if np.isinf(delta):
        for _ in range(int(niter)):
            w = A @ u_prev
            nw = np.linalg.norm(w, 2)
            if nw == 0:
                return 0.0, u_prev
            u = w / nw

            if np.linalg.norm(u - u_prev, ord=p) < eps:
                lam = float(u.T @ (A @ u))
                return lam, u

            u_prev = u

        lam = float(u_prev.T @ (A @ u_prev))
        return lam, u_prev

    B = A - float(delta) * np.eye(n)

    try:
        lu, piv = spla.lu_factor(B)
    except Exception:
        return float(delta), None

    Udiag = np.diag(lu)
    if np.any(np.isclose(Udiag, 0.0)):
        return float(delta), None

    for _ in range(int(niter)):
        w = spla.lu_solve((lu, piv), u_prev)

        nw = np.linalg.norm(w, 2)
        if nw == 0:
            return float(delta), None
        u = w / nw

        if np.linalg.norm(u - u_prev, ord=p) < eps:
            mu = float(u.T @ spla.lu_solve((lu, piv), u))
            if mu == 0:
                return float(delta), u
            lam = float(delta) + 1.0 / mu
            return lam, u

        u_prev = u

    mu = float(u_prev.T @ spla.lu_solve((lu, piv), u_prev))
    if mu == 0:
        return float(delta), u_prev
    lam = float(delta) + 1.0 / mu
    return lam, u_prev

def approxmc1c(base, ab, funcion, npts=200):
    """
    Resuelve el ajuste continuo por mínimos cuadrados en [a,b] con ecuaciones normales

    Entrada:
        base : lista {φ_1, ..., φ_n}.
        ab : intervalo [a,b].
        funcion : función f(x).
        npts : puntos de Gauss-Legendre para aproximar integrales.

    Salida:
        c : coeficientes del ajuste en la base.
        Ecm : error cuadrático medio.
        r2 : coeficiente de determinación.
    """
    a, b = ab
    base = list(base)

    t, w = leggauss(int(npts))              
    x = 0.5 * (b - a) * t + 0.5 * (a + b)   
    wx = 0.5 * (b - a) * w

    fx = np.asarray(funcion(x), dtype=float)
    Phi = np.vstack([np.asarray(phi(x), dtype=float) for phi in base])

    G = Phi @ (wx * Phi).T
    d = Phi @ (wx * fx)
    c = np.linalg.solve(G, d)

    ff = float(np.sum(wx * fx * fx))                 
    Ec = ff - 2.0 * float(c @ d) + float(c @ (G @ c))
    Ecm = Ec / (b - a)

    fmean = float(np.sum(wx * fx) / (b - a))
    Sy2 = float(np.sum(wx * (fx - fmean) ** 2))
    r2 = 1.0 - (Ec / Sy2) if Sy2 > 0 else np.nan

    return c, Ecm, r2

def approxmc1d(base, x, y):
    """
    Calcula el ajuste por mínimos cuadrados discreto usando ecuaciones normales

    Entrada:
        base : lista {φ_1, ..., φ_n}.
        x : puntos del soporte.
        y : valores en los puntos.

    Salida:
        c : vector con los coeficientes en la base.
        Ecm : error cuadrático medio (Ec/N).
        r2 : coeficiente de determinación.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = list(base)
    n = len(base)
    N = len(x)

    Phi = np.vstack([phi(x) for phi in base])  

    G = Phi @ Phi.T                           
    d = Phi @ y                               
    c = np.linalg.solve(G, d)

    yy = float(y @ y)                        
    Ec = yy - 2.0 * float(c @ d) + float(c @ (G @ c))
    Ecm = Ec / N

    ymean = float(np.mean(y))
    Sy2 = float(np.sum((y - ymean) ** 2))
    r2 = 1.0 - (Ec / Sy2) if Sy2 > 0 else np.nan

    return c, Ecm, r2

def approxmc1d_eval(base, coef, z):
    """
    Evalúa la función de aproximación en los puntos z.

    Entrada:
        base : lista {φ_1, ..., φ_n}.
        coef : coeficientes c del ajuste.
        z : puntos donde evaluar.

    Salida:
        psi : vector con los valores de la función aproximación.
    """
    base = list(base)
    c = np.asarray(coef, dtype=float)
    z = np.asarray(z, dtype=float)

    if len(base) != len(c):
        raise ValueError("base y coef deben tener la misma longitud.")

    psi = np.zeros_like(z, dtype=float)
    for ci, phi in zip(c, base):
        psi += ci * np.asarray(phi(z), dtype=float)

    return psi

def poly_optimo(x, y):
    """
    Calcula el polinomio de ajuste por mínimos cuadrados (grados crecientes)
    y selecciona el de menor error estándar.

    Entrada:
        x : puntos del soporte.
        y : valores en los puntos.

    Salida:
        p_opt : coeficientes del polinomio óptimo (orden creciente).
        errores : vector con los errores estándar para cada grado.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)

    errores = []
    polinomios = []

    # grados desde 0 hasta N-1
    for grado in range(N):
        # Ajuste polinómico por mínimos cuadrados
        coef = np.polyfit(x, y, grado)
        polinomios.append(coef)

        # Evaluación y error estándar
        y_aprox = np.polyval(coef, x)
        Ec = np.sum((y - y_aprox) ** 2)
        E_std = np.sqrt(Ec / (N - grado - 1)) if N - grado - 1 > 0 else np.inf
        errores.append(E_std)

    errores = np.array(errores)
    idx = np.argmin(errores)

    return polinomios[idx], errores