#!/usr/bin/env python3
"""
metodos_raices_todas_ecuaciones.py
Implementa Bisección, Newton-Raphson y Secante para 4 ecuaciones predefinidas.
Genera CSVs con las corridas y gráficos (matplotlib) mostrando la función y las iteraciones.
"""

import math, csv, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("resultados_python")
OUTDIR.mkdir(exist_ok=True)

# --- Definición de funciones y derivadas ---
def f1(x): return x**3 - math.exp(0.8*x) - 20.0
def df1(x): return 3.0*x**2 - 0.8*math.exp(0.8*x)

def f2(x): return 3.0*math.sin(0.5*x) - 0.5*x + 2.0
def df2(x): return 1.5*math.cos(0.5*x) - 0.5

def f3(x): return x**3 - x**2*math.exp(-0.5*x) - 3.0*x + 1.0
def df3(x):
    # derivada: 3x^2 - 2x e^{-0.5x} + 0.5 x^2 e^{-0.5x} - 3
    return 3.0*x**2 - 2.0*x*math.exp(-0.5*x) + 0.5*x**2*math.exp(-0.5*x) - 3.0

def f4(x): return math.cos(x)**2 - 0.5*x*math.exp(0.3*x) + 5.0
def df4(x): 
    # d/dx cos^2(x) = 2 cos x * (-sin x) = - sin(2x)
    # d/dx [-0.5 x e^{0.3x}] = -0.5 e^{0.3x} -0.5*0.3 x e^{0.3x}
    return -math.sin(2.0*x) - 0.5*math.exp(0.3*x)*(1.0 + 0.3*x)

FUNCS = [
    ("f1_x3_e08x_minus20", f1, df1, (0.0, 8.0)),
    ("f2_3sin05x_minus05x_plus2", f2, df2, (-10.0, 10.0)),
    ("f3_x3_x2exp_minus05x_minus3x_plus1", f3, df3, (-5.0, 5.0)),
    ("f4_cos2x_minus05xexp03x_plus5", f4, df4, (0.0, 10.0))  # buscar positivas
]

# --- Métodos genéricos ---
def biseccion(f, a, b, tol=1e-8, maxiter=200):
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")
    rows = []
    x_prev = None
    for it in range(1, maxiter+1):
        c = 0.5*(a+b)
        fc = f(c)
        err = abs(c - x_prev) if x_prev is not None else None
        rows.append((it, a, b, c, fa, fb, fc, err if err is not None else 0.0))
        if abs(fc) < tol or (err is not None and err < tol):
            break
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        x_prev = c
    return c, fc, rows

def newton(f, df, x0, tol=1e-10, maxiter=100):
    rows = []
    x = x0
    for k in range(1, maxiter+1):
        fx = f(x); dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivada casi cero en Newton.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        rows.append((k, x, fx, dfx, x_new, err))
        x = x_new
        if abs(fx) < tol or err < tol:
            break
    return x, f(x), rows

def secante(f, x0, x1, tol=1e-10, maxiter=200):
    rows = []
    x_prev, x = x0, x1
    fx_prev, fx = f(x_prev), f(x)
    for k in range(1, maxiter+1):
        denom = (fx - fx_prev)
        if abs(denom) < 1e-14:
            raise ZeroDivisionError("Denominador pequeño en Secante.")
        x_new = x - fx*(x - x_prev)/denom
        err = abs(x_new - x)
        rows.append((k, x_prev, x, fx_prev, fx, x_new, err))
        x_prev, fx_prev = x, fx
        x, fx = x_new, f(x_new)
        if abs(fx) < tol or err < tol:
            break
    return x, fx, rows

# --- util: guardar CSV ---
def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

# --- util: graficar función y puntos de iteración ---
def plot_with_iters(name, f, rows_bis, rows_new, rows_sec, interval, outdir):
    a, b = interval
    xs = np.linspace(a, b, 1000)
    ys = [f(x) for x in xs]
    plt.figure(figsize=(10,6))
    plt.plot(xs, ys, label='f(x)')
    plt.axhline(0, color='k', linewidth=0.8)
    # iteraciones en eje x (y=0)
    if rows_bis:
        xb = [r[3] for r in rows_bis]
        plt.scatter(xb, [0]*len(xb), marker='o', label='Bisección iters', alpha=0.6)
    if rows_new:
        xn = [r[4] for r in rows_new]
        plt.scatter(xn, [0]*len(xn), marker='x', label='Newton iters', alpha=0.6)
    if rows_sec:
        xsr = [r[5] for r in rows_sec]
        plt.scatter(xsr, [0]*len(xsr), marker='s', label='Secante iters', alpha=0.6)
    plt.xlim(a, b)
    ymin, ymax = min(ys), max(ys)
    plt.ylim(ymin*1.1, ymax*1.1)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    p = outdir / f"grafico_{name}.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Guardado gráfico: {p}")

# --- Ejecutar para todas ---
def procesar_todas():
    for (name, f, df, interval) in FUNCS:
        print(f"\nProcesando {name} ...")
        a, b = interval
        # Intenta encontrar cambio de signo para cada función si extremo no sirve (simple barrido)
        try:
            # Bisección: buscar un sub-intervalo con signo opuesto si el intervalo dado no tiene
            fa, fb = f(a), f(b)
            if fa*fb > 0:
                # Barrido simple para encontrar un par con signo opuesto en [a,b]
                found = False
                N=200
                xs = np.linspace(a,b,N)
                for i in range(N-1):
                    if f(xs[i])*f(xs[i+1])<0:
                        a0, b0 = xs[i], xs[i+1]
                        found = True
                        break
                if not found:
                    print(f"  Atención: no se encontró cambio de signo en {interval} para {name}. Se saltará Bisección.")
                    rows_bis = []
                    root_bis = None
                else:
                    try:
                        root_bis, froot_bis, rows_bis = biseccion(f, a0, b0, tol=1e-8, maxiter=200)
                    except Exception as e:
                        print("  Error bisección:", e); rows_bis=[]
                        root_bis = None
            else:
                root_bis, froot_bis, rows_bis = biseccion(f, a, b, tol=1e-8, maxiter=200)
        except Exception as e:
            print("  Bisección falló:", e); rows_bis=[]; root_bis=None

        # Newton: elegir x0 centro del intervalo
        try:
            x0 = 0.5*(a+b)
            root_new, froot_new, rows_new = newton(f, df, x0, tol=1e-12, maxiter=200)
        except Exception as e:
            print("  Newton falló:", e); rows_new=[]; root_new=None

        # Secante: usar extremos del intervalo (o pequeñas variaciones)
        try:
            root_sec, froot_sec, rows_sec = secante(f, a+1e-3, b-1e-3, tol=1e-12, maxiter=300)
        except Exception as e:
            print("  Secante falló:", e); rows_sec=[]; root_sec=None

        # Guardar CSVs
        if rows_bis:
            save_csv(OUTDIR / f"biseccion_{name}.csv",
                     ['iter','a','b','c','f(a)','f(b)','f(c)','error'], rows_bis)
        if rows_new:
            save_csv(OUTDIR / f"newton_{name}.csv",
                     ['iter','x','f(x)','fprime(x)','x_next','error'], rows_new)
        if rows_sec:
            save_csv(OUTDIR / f"secante_{name}.csv",
                     ['iter','x_prev','x','f(x_prev)','f(x)','x_next','error'], rows_sec)

        # Graficar
        try:
            plot_with_iters(name, f,
                            rows_bis if rows_bis else None,
                            rows_new if rows_new else None,
                            rows_sec if rows_sec else None,
                            interval, OUTDIR)
        except Exception as e:
            print("  Falló graficar:", e)

        # Resumen por pantalla
        print("  Resumen:")
        print("    Bisección root:", root_bis if 'root_bis' in locals() else None, "iters:", len(rows_bis) if rows_bis else 0)
        print("    Newton    root:", root_new if 'root_new' in locals() else None, "iters:", len(rows_new) if rows_new else 0)
        print("    Secante   root:", root_sec if 'root_sec' in locals() else None, "iters:", len(rows_sec) if rows_sec else 0)

if __name__ == "__main__":
    procesar_todas()
    print("\nTerminado. CSVs y gráficos en carpeta:", OUTDIR)
