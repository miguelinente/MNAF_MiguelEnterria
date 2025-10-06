# -*- coding: utf-8 -*-
''' Archivo que contiene todas la funciones requeridas por los ejercicios '''

from __init__ import _hello
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def euclides(D, d):
    """
    Calcula el máximo común divisor (MCD) de dos enteros D y d
    utilizando el algoritmo de Euclides clásico.

    Parámetros
    ----------
    D : int
        Primer entero (dividendo inicial).
    d : int
        Segundo entero (divisor inicial).

    Retorna
    -------
    c : int
        Máximo común divisor de D y d.

    Algoritmo
    ---------
    1. Calcular cociente y resto de la división entera: D = d * c + r
    2. Si r = 0, el MCD es d
    3. Si r ≠ 0, sustituir (D, d) por (d, r) y volver a 1
    """

    _hello()

    # Paso 1: calcular resto inicial
    r = D % d

    # Paso 2: repetir mientras el resto no sea cero
    while r != 0:
        # Sustituir: el nuevo dividendo es d, el nuevo divisor es r
        D, d = d, r
        # Recalcular el resto
        r = D % d

    # Paso 3: cuando r = 0, el MCD es d
    return d

def espiral_esfera(r=1.0, c=(1.0, 1.0, 1.0), npts=800, mostrar_esfera=True, ax=None):
    """
    Representa una espiral sobre la esfera (x-cx)^2+(y-cy)^2+(z-cz)^2 = r^2
    usando la parametrización del guion:

        x = cx + R*cos(4*pi*α)
        y = cy + R*sin(4*pi*α)
        z = α
        R = sqrt(r^2 - (z - cz)^2),   cz - r ≤ α ≤ cz + r

    Parámetros
    ----------
    r : float
        Radio de la esfera. (por defecto 1.0)
    c : tuple[float, float, float]
        Centro de la esfera (cx, cy, cz). (por defecto (1,1,1))
    npts : int
        Número de puntos a muestrear a lo largo de la espiral.
    mostrar_esfera : bool
        Si True, dibuja también la esfera para referencia.
    ax : matplotlib.axes._subplots.Axes3DSubplot | None
        Ejes 3D donde dibujar. Si None, crea una figura nueva.

    Retorna
    -------
    x, y, z : np.ndarray
        Coordenadas de la espiral generada.
    """
    cx, cy, cz = c
    # α recorre el “alto” de la esfera de cz-r a cz+r
    alpha = np.linspace(cz - r, cz + r, npts)
    # Radio de la circunferencia horizontal a altura z = α
    R = np.sqrt(np.maximum(0.0, r**2 - (alpha - cz)**2))
    # Ángulo como en el enunciado: 4π·α  (genera 4 vueltas si cz=1, r=1 → α∈[0,2])
    theta = 4.0 * np.pi * alpha

    x = cx + R * np.cos(theta)
    y = cy + R * np.sin(theta)
    z = alpha

    # Dibujo
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection='3d')

    ax.plot(x, y, z, lw=2, label="Espiral")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Espiral sobre esfera")

    if mostrar_esfera:
        # mallado de la esfera para contexto visual
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, alpha=0.3, linewidth=0.7)

    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # relación 1:1:1
    plt.tight_layout()
    plt.show()

    return x, y, z

def elipsoide(a=(3.0, 2.0, 1.5), c=(1.0, 0.0, -0.5),
              n_alpha=120, n_beta=60, wire=False, ax=None):
    """
    Representa el elipsoide:
        ((x-cx)/ax)^2 + ((y-cy)/ay)^2 + ((z-cz)/az)^2 = 1
    con la parametrización del guion:
        x = cx + ax*cos(α)*cos(β)
        y = cy + ay*cos(α)*sin(β)
        z = cz + az*sin(α)
        -π ≤ α ≤ π,  0 ≤ β ≤ π

    Parámetros
    ----------
    a : tuple(float,float,float)
        Semiejes (ax, ay, az).  Por defecto (3, 2, 1.5).
    c : tuple(float,float,float)
        Centro (cx, cy, cz). Por defecto (1, 0, -0.5).
    n_alpha, n_beta : int
        Puntos de muestreo para α y β.
    wire : bool
        Si True dibuja malla (wireframe). Si False superficie sólida.
    ax : Axes3D | None
        Ejes 3D donde dibujar. Si None se crea figura nueva.

    Retorna
    -------
    X, Y, Z : np.ndarray
        Mallas con las coordenadas del elipsoide.
    """
    ax_e, ay_e, az_e = a
    cx, cy, cz = c

    alpha = np.linspace(-np.pi, np.pi, n_alpha)   # ángulo “vertical”
    beta  = np.linspace(0.0, np.pi, n_beta)       # ángulo “horizontal”
    A, B  = np.meshgrid(alpha, beta, indexing="ij")

    X = cx + ax_e * np.cos(A) * np.cos(B)
    Y = cy + ay_e * np.cos(A) * np.sin(B)
    Z = cz + az_e * np.sin(A)

    # Dibujo
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, projection='3d')

    if wire:
        ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, linewidth=0.7, alpha=0.9)
    else:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7)

    ax.set_title("Elipsoide")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
    return X, Y, Z

# Bloque de pruebas
if __name__ == "__main__":
    elipsoide()