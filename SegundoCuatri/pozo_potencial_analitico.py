import numpy as np
from scipy.optimize import brentq
import scipy.constants as const

# ─── Constantes físicas ────────────────────────────────────────────────────
hbar_eVs   = 6.582119569e-16   # hbar en eV·s
hbar_Js    = const.hbar         # hbar en J·s
m_e_eV     = 0.51099895e6      # masa electrón en eV/c²
c_m_s      = const.c            # velocidad de la luz en m/s
m_e_kg     = const.m_e          # masa electrón en kg

# ─── Parámetros del pozo ───────────────────────────────────────────────────
V0_eV = 244.0           # profundidad del pozo en eV
a_m   = 1.0e-10         # anchura del pozo en metros (1 Å)

# ─── Cálculo de k = 2·m·a²·V0 / hbar² ────────────────────────────────────
# Usamos unidades SI para k (sin dimensiones en unidades del guión)
k = (2 * m_e_kg * a_m**2 * V0_eV * const.eV) / hbar_Js**2

print("=" * 65)
print("  POZO DE POTENCIAL CUADRADO FINITO — SOLUCIÓN ANALÍTICA")
print("  Eisberg & Resnick, Apéndice H")
print("=" * 65)
print(f"\n  Parámetros:")
print(f"    V0     = {V0_eV}  eV")
print(f"    a      = {a_m*1e10:.1f}  Å  = {a_m:.2e} m")
print(f"    m_e    = {m_e_kg:.6e}  kg")
print(f"    hbar   = {hbar_Js:.6e}  J·s")
print(f"    k      = 2·m·a²·V0/ℏ²  =  {k:.6f}  (adimensional)")
print(f"    sqrt(k)= {np.sqrt(k):.6f}")

# ─── Ecuaciones trascendentes ──────────────────────────────────────────────
# f_even(α) = 0  para estados pares
# f_odd(α)  = 0  para estados impares

def f_even(alpha):
    """
    Estados pares: tan(sqrt(k*α)/2) - sqrt((1-α)/α) = 0
    Equivalente a: sqrt(α)·tan(sqrt(k·α)/2) - sqrt(1-α) = 0
    """
    if alpha <= 0 or alpha >= 1:
        return np.nan
    arg = np.sqrt(k * alpha) / 2.0
    return np.tan(arg) - np.sqrt((1.0 - alpha) / alpha)

def f_odd(alpha):
    """
    Estados impares: -cot(sqrt(k·α)/2) - sqrt((1-α)/α) = 0
    Equivalente a: tan(sqrt(k·α)/2) + sqrt(α/(1-α)) = 0
    No usar cot directamente: usar tan + condición
    -cos/sin = sqrt((1-α)/α)  ⟺  -cot(θ) = sqrt((1-α)/α)
    """
    if alpha <= 0 or alpha >= 1:
        return np.nan
    arg = np.sqrt(k * alpha) / 2.0
    # -cot(arg) = cos(arg)/(-sin(arg))
    return -1.0 / np.tan(arg) - np.sqrt((1.0 - alpha) / alpha)

# ─── Búsqueda de raíces por barrido + bisección ───────────────────────────
# La función f_even/f_odd tiene discontinuidades (tangente)
# Buscamos cambios de signo REALES (no cruzar por ±inf de la tangente)

def find_roots(f, n_pts=200_000, eps=1e-12):
    """
    Barrido de α en (eps, 1-eps) buscando cambios de signo reales.
    Descarta cruces debidos a discontinuidades de la tangente.
    """
    alphas = np.linspace(eps, 1.0 - eps, n_pts)
    roots = []
    
    fvals = np.array([f(a) for a in alphas])
    
    for i in range(len(alphas) - 1):
        f0, f1 = fvals[i], fvals[i+1]
        if np.isnan(f0) or np.isnan(f1):
            continue
        if f0 * f1 < 0:
            # Distinguir cruce real de discontinuidad:
            # En una discontinuidad la función salta de -inf a +inf (o viceversa)
            # sin pasar por cero suavemente.
            # Criterio: si |f0| y |f1| son AMBOS grandes, es discontinuidad.
            if abs(f0) > 50 and abs(f1) > 50:
                continue  # probable discontinuidad, no raíz real
            try:
                root = brentq(f, alphas[i], alphas[i+1], xtol=1e-14, rtol=1e-14, maxiter=500)
                roots.append(root)
            except ValueError:
                pass
    return roots

print("\n─── Búsqueda de estados ligados ─────────────────────────────────")

roots_even = find_roots(f_even)
roots_odd  = find_roots(f_odd)

# Mezclar y ordenar por energía
all_states = []
for alpha in roots_even:
    all_states.append(("par",  alpha, alpha * V0_eV))
for alpha in roots_odd:
    all_states.append(("impar", alpha, alpha * V0_eV))

all_states.sort(key=lambda x: x[2])

print(f"\n  Total de estados ligados encontrados: {len(all_states)}")
print()
print(f"  {'n':>3}  {'Paridad':>6}  {'α = E/V0':>18}  {'E (eV)':>14}  {'E/V0 (%)':>10}")
print(f"  {'-'*3}  {'-'*6}  {'-'*18}  {'-'*14}  {'-'*10}")

for n, (parity, alpha, E) in enumerate(all_states, 1):
    print(f"  {n:>3}  {parity:>6}  {alpha:>18.14f}  {E:>14.8f}  {100*alpha:>9.5f}%")

# ─── Verificación de las ecuaciones trascendentes ─────────────────────────
print()
print("─── Verificación (residuos de las ec. trascendentes) ────────────")
print()
for n, (parity, alpha, E) in enumerate(all_states, 1):
    if parity == "par":
        residuo = f_even(alpha)
        eq_name = "tan(√(kα)/2) − √((1−α)/α)"
    else:
        residuo = f_odd(alpha)
        eq_name = "−cot(√(kα)/2) − √((1−α)/α)"
    print(f"  n={n} ({parity:>6}): α={alpha:.14f}, residuo={residuo:.2e}  ✓" if abs(residuo) < 1e-10
          else f"  n={n} ({parity:>6}): α={alpha:.14f}, residuo={residuo:.2e}")

# ─── Cálculo de l y kappa para cada estado ────────────────────────────────
print()
print("─── Números de onda y constante de decaimiento ──────────────────")
print()
print(f"  {'n':>3}  {'l·a/2':>12}  {'κ·a/2':>12}  {'l (Å⁻¹)':>12}  {'κ (Å⁻¹)':>12}")
print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

for n, (parity, alpha, E) in enumerate(all_states, 1):
    la_half = np.sqrt(k * alpha) / 2.0           # l·a/2 (adimensional)
    ka_half = np.sqrt(k * (1 - alpha)) / 2.0     # κ·a/2 (adimensional)
    l_inv_A  = np.sqrt(k * alpha) / (a_m * 1e10) # l en Å⁻¹
    kap_inv_A= np.sqrt(k * (1 - alpha)) / (a_m * 1e10)  # κ en Å⁻¹
    print(f"  {n:>3}  {la_half:>12.8f}  {ka_half:>12.8f}  {l_inv_A:>12.6f}  {kap_inv_A:>12.6f}")

# ─── Resumen final con alta precisión ─────────────────────────────────────
print()
print("=" * 65)
print("  RESUMEN: valores de α con máxima precisión")
print("=" * 65)
print()
for n, (parity, alpha, E) in enumerate(all_states, 1):
    print(f"  Estado n={n} ({parity:>6}):  α = {alpha:.15e}")
    print(f"                         E = {E:.10f} eV")
    print()

print("─── Comparación con pozo infinito (referencia) ──────────────────")
print()
# Pozo infinito: E_n = n²·π²·ℏ²/(2m·a²)
for n in range(1, len(all_states)+1):
    E_inf = (n * np.pi)**2 * hbar_Js**2 / (2 * m_e_kg * a_m**2) / const.eV
    alpha_inf = E_inf / V0_eV
    print(f"  n={n}: E_inf = {E_inf:.4f} eV  (α_inf = {alpha_inf:.6f})")
    if n <= len(all_states):
        _, alpha_fin, E_fin = all_states[n-1]
        print(f"        E_fin = {E_fin:.4f} eV  (α_fin = {alpha_fin:.6f})")
        print(f"        Diferencia: ΔE = {E_inf - E_fin:.4f} eV  ({100*(E_inf-E_fin)/E_inf:.2f}%)")
        print()
