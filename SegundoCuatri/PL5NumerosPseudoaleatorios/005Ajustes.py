import numpy as np
from scipy.optimize import curve_fit

def modelo_bi(t, N0, tau):
    return N0 * np.exp(-t / tau)

def modelo_po(t, N0, tBi, tPo):
    w_Bi, w_Po = 1.0 / tBi, 1.0 / tPo
    return (w_Bi / (w_Po - w_Bi)) * N0 * (np.exp(-w_Bi * t) - np.exp(-w_Po * t))

def ejecutar_analisis_completo():
    poblaciones = [100, 1000, 10000, 100000, 1000000]
    tau_bi_real, tau_po_real = 7.5, 190.0
    t_sim = np.linspace(0, 500, 1000)

    print(f"{'N0':<8} | {'Isótopo':<10} | {'tau_fit':<10} ")
    print("-" * 55)

    for n in poblaciones:
        # --- Simulación Monte Carlo ---
        t_nace_po = -tau_bi_real * np.log(np.random.uniform(0, 1, n))
        t_muere_po = t_nace_po - tau_po_real * np.log(np.random.uniform(0, 1, n))
        
        n_bi_sim = np.array([np.sum(t_nace_po > t) for t in t_sim])
        n_po_sim = np.array([np.sum((t_nace_po <= t) & (t_muere_po > t)) for t in t_sim])

        # --- Ajuste Bismuto ---
        popt_bi, pcov_bi = curve_fit(modelo_bi, t_sim[:100], n_bi_sim[:100], p0=[n, 7.5])
        tau_bi_f = popt_bi[1]
        err_bi = np.sqrt(pcov_bi[1,1])
        rel_bi = abs(tau_bi_f - tau_bi_real) / tau_bi_real * 100

        # --- Ajuste Polonio ---
        popt_po, pcov_po = curve_fit(modelo_po, t_sim, n_po_sim, p0=[n, 7.5, 190])
        tau_po_f = popt_po[2]
        err_po = np.sqrt(pcov_po[2,2])
        rel_po = abs(tau_po_f - tau_po_real) / tau_po_real * 100

        print(f"{n:<8} | Bismuto   | {tau_bi_f:>10.3f}")
        print(f"{n:<8} | Polonio   | {tau_po_f:>10.3f}")

if __name__ == "__main__":
    ejecutar_analisis_completo()