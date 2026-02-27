# app.py
# Dominio N–Mx–My + curve chiuse Mx–N e My–N per sezione rettangolare in c.a.
#
# - Dominio 3D: modello rapido con blocchetto uniforme (σc = ηc·fcd, cls solo compressione)
# - M–N (Mx–N; My–N): inviluppo chiuso tramite integrazione a fibre EC2 (parabola‑rettangolo) su griglia (εm, κ)
#   -> N in ascissa, M in ordinata; copre N>0 e N<0 (compressione e trazione), M ±
#
# Convenzioni:
#   N > 0 = compressione
#   Mx = Σ(y·F) ; My = Σ(-x·F) ; F>0 a compressione
#
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os
import csv
import io

#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()


#    === Start application ===
if 'datiGeo' not in st.session_state:
    st.session_state['datiGeo'] = []
if "base" not in st.session_state:
    st.session_state["base"] = 300.0  # default width
if "height" not in st.session_state:
    st.session_state["height"] = 500.0  # default height
if"cover" not in st.session_state:
    st.session_state["cover"] = 30.0  # default cover
if "stirrups_dia" not in st.session_state:
    st.session_state["stirrups_dia"] = 8  # default stirrups diameter

if "layers" not in st.session_state:
    st.session_state["layers"] = []


# ===================== Utility geometriche su poligoni =====================

def polygon_area_and_centroid(poly):
    """Area (>0) e baricentro (cx, cy) di un poligono CCW."""
    if len(poly) < 3:
        return 0.0, (0.0, 0.0)
    x = np.array([p[0] for p in poly] + [poly[0][0]])
    y = np.array([p[1] for p in poly] + [poly[0][1]])
    cross = x[:-1]*y[1:] - x[1:]*y[:-1]
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-12:
        return 0.0, (0.0, 0.0)
    Cx = (1/(6*A)) * np.sum((x[:-1] + x[1:]) * cross)
    Cy = (1/(6*A)) * np.sum((y[:-1] + y[1:]) * cross)
    return abs(A), (Cx, Cy)

def intersect_segment_with_line(p1, p2, n, rho):
    """Intersezione del segmento p1->p2 con la retta n·p - rho = 0."""
    p1 = np.array(p1); p2 = np.array(p2); n = np.array(n, dtype=float)
    d1 = float(np.dot(n, p1) - rho)
    d2 = float(np.dot(n, p2) - rho)
    denom = (d1 - d2)
    if abs(denom) < 1e-14:
        return tuple(p1)
    t = d1 / (d1 - d2)
    P = p1 + t * (p2 - p1)
    return (float(P[0]), float(P[1]))

def clip_polygon_with_halfplane(poly, n, rho):
    """Sutherland-Hodgman: interseca poligono con semipiano {p | n·p - rho <= 0}."""
    if len(poly) == 0:
        return []
    output = []
    prev = poly[-1]
    d_prev = float(np.dot(n, prev) - rho)
    for curr in poly:
        d_curr = float(np.dot(n, curr) - rho)
        curr_in = (d_curr <= 1e-12)
        prev_in = (d_prev <= 1e-12)
        if curr_in:
            if not prev_in:
                inter = intersect_segment_with_line(prev, curr, n, rho)
                output.append(inter)
            output.append(tuple(curr))
        else:
            if prev_in:
                inter = intersect_segment_with_line(prev, curr, n, rho)
                output.append(inter)
        prev, d_prev = curr, d_curr
    return output

# ===================== Sezione rettangolare e armature =====================

def rect_section(b, h):
    """Rettangolo centrato nell'origine, lati paralleli agli assi, CCW."""
    bx = b/2.0; hy = h/2.0
    return [(-bx, -hy), (bx, -hy), (bx, hy), (-bx, hy)]

def rebars_from_layers(csv_path, section_width, section_height):
    """
    Legge un file CSV con strati di armatura e restituisce:
    - lista di tuple (x, y, phi, As)
    dove:
        x, y = coordinate della barra
        phi  = diametro della barra
        As   = area della barra
    section_height serve per convertire y_layer (misurato dall'alto)
    in coordinate centrate sulla sezione.
    """
    rebar_pts = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(float(row["num_bars"]))  
            phi = float(row["dia"])
            y_layer_top = float(row["y_layer"])
            x_first = float(row["x_first"])
            spacing = float(row["rebar_spacing"])

            # Area singola barra
            As = math.pi * (phi**2) / 4.0

            # Converti y_layer (misurato dall'alto) in coordinate centrate
            y = section_height/2 - y_layer_top      # centra l'asse x
            x_offset = section_width / 2.0          # centra l'asse y

            # Genera le barre dello strato
            for i in range(n):
                x_raw = x_first + i * spacing
                x = x_raw - x_offset # centratura
                rebar_pts.append((x, y, phi, As))

    return rebar_pts

# ===================== Forze interne (blocco uniforme, per dominio 3D) =====================

def section_projection_extrema(poly, n):
    dots = [np.dot(n, np.array(p)) for p in poly]
    return float(min(dots)), float(max(dots))

def compute_internal_forces(poly, rebars, mat, theta, c):
    """
    Modello blocchetto uniforme (σc = ηc·fcd, cls solo compressione).
    Ritorna N[kN], Mx[kNm], My[kNm].
    """
    n = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    mn, mx = section_projection_extrema(poly, n)
    rho = mn + c
    comp_poly = clip_polygon_with_halfplane(poly, n, rho)
    Acomp, (xc, yc) = polygon_area_and_centroid(comp_poly)

    sigma_c = mat['eta_c'] * mat['fcd']     # MPa = N/mm^2
    Fc = sigma_c * Acomp                    # N (+ compressione)
    Mx_c = yc * Fc
    My_c = -xc * Fc

    N_s = 0.0; Mx_s = 0.0; My_s = 0.0
    Es = mat['Es']; fyd = mat['fyd']; eps_cu = mat['eps_cu']

    for (x,y,phi,As) in rebars:
        d = float(np.dot(n, np.array([x,y])) - rho)         # distanza dalla linea neutra
        eps = (eps_cu * d / (-c)) if c > 1e-12 else -1e-9   # deformazione acciaio
        sig = max(-fyd, min(fyd, Es * eps))                 # tensione acciaio (limitata a ±fyd)
        Fs = sig * As                                       # forza della singola barra
        N_s += Fs; Mx_s += y * Fs; My_s += -x * Fs

    N_tot = (Fc + N_s)/1000.0
    Mx_tot = (Mx_c + Mx_s)/1e6
    My_tot = (My_c + My_s)/1e6
    return N_tot, Mx_tot, My_tot

def sample_domain_grid(poly, rebars, mat, n_theta=72, n_c=40): 
    """
    θ in [0, 2π) per coprire segni di Mx/My.
    """
    thetas = np.linspace(0.0, 2*math.pi, n_theta, endpoint=False)
    Ns = np.zeros((n_theta, n_c+1))
    Mxs = np.zeros_like(Ns)
    Mys = np.zeros_like(Ns)
    cs_grid = np.zeros_like(Ns)
    Ls = np.zeros(n_theta)
    for i, theta in enumerate(thetas):
        n = np.array([math.cos(theta), math.sin(theta)])
        mn, mx = section_projection_extrema(poly, n)
        Lproj = max(0.0, mx - mn)
        Ls[i] = Lproj
        for j in range(n_c+1):
            c = Lproj * j / n_c
            N, Mx, My = compute_internal_forces(poly, rebars, mat, theta, c)
            Ns[i,j] = N; Mxs[i,j] = Mx; Mys[i,j] = My; cs_grid[i,j] = c
    return thetas, cs_grid, Ns, Mxs, Mys, Ls

def grid_to_mesh_indices(n_theta, n_c, wrap_theta=True):
    I = []; J = []; K = []
    for i in range(n_theta):
        inext = (i+1) % n_theta if wrap_theta else min(i+1, n_theta-1)
        for j in range(n_c):
            a = i*(n_c+1) + j
            b = inext*(n_c+1) + j
            c = inext*(n_c+1) + (j+1)
            d = i*(n_c+1) + (j+1)
            I += [a, a]; J += [b, c]; K += [c, d]
    return I, J, K


# ===================== Sezione Mx–My a N = cost (chiusa) =====================

def slice_at_N_target_closed(poly, rebars, mat, N_target, thetas, tol=2.0, max_iter=30):
    """Contorno chiuso Mx-My alla quota N_target."""
    pts = []
    for theta in thetas:
        n = np.array([math.cos(theta), math.sin(theta)])
        mn, mx = section_projection_extrema(poly, n)
        Lproj = max(0.0, mx - mn)
        c_lo, c_hi = 0.0, Lproj
        N_lo, Mx_lo, My_lo = compute_internal_forces(poly, rebars, mat, theta, c_lo)
        N_hi, Mx_hi, My_hi = compute_internal_forces(poly, rebars, mat, theta, c_hi)

        if not (min(N_lo, N_hi) - tol <= N_target <= max(N_lo, N_hi) + tol):
            cand = [(abs(N_lo - N_target), Mx_lo, My_lo),
                    (abs(N_hi - N_target), Mx_hi, My_hi)]
            _, Mx_b, My_b = min(cand, key=lambda t: t[0])
            pts.append((Mx_b, My_b))
            continue

        # bisezione su c
        for _ in range(max_iter):
            c_mid = 0.5*(c_lo + c_hi)
            N_mid, Mx_mid, My_mid = compute_internal_forces(poly, rebars, mat, theta, c_mid)
            if abs(N_mid - N_target) <= tol:
                pts.append((Mx_mid, My_mid))
                break
            if (N_lo - N_target)*(N_mid - N_target) <= 0:
                c_hi = c_mid; N_hi = N_mid
            else:
                c_lo = c_mid; N_lo = N_mid
        else:
            # fallback
            cand = []
            for c_try in (c_lo, c_hi):
                N_try, Mx_try, My_try = compute_internal_forces(poly, rebars, mat, theta, c_try)
                cand.append((abs(N_try-N_target), Mx_try, My_try))
            _, Mx_b, My_b = min(cand, key=lambda t: t[0])
            pts.append((Mx_b, My_b))

    pts = np.array(pts)
    ang = np.arctan2(pts[:,1], pts[:,0])   # atan2(My, Mx)
    order = np.argsort(ang)
    Mx_ord = pts[order,0]
    My_ord = pts[order,1]
    Mx_closed = np.concatenate([Mx_ord, Mx_ord[:1]])
    My_closed = np.concatenate([My_ord, My_ord[:1]])
    return Mx_closed, My_closed


# ===================== Bending capacities at given N =====================
def bending_capacities_at_N(N_hull, M_hull, N_target):
    """
    Ricava M_Rd+ e M_Rd- per un dato N_target
    dall'inviluppo (N_hull, M_hull) - curva chiusa.
    """
    N_vals = np.array(N_hull)
    M_vals = np.array(M_hull)
    n = len(N_vals)

    Ms_at_N = []

    for i in range(n-1):
        N1, N2 = N_vals[i], N_vals[i+1]
        M1, M2 = M_vals[i], M_vals[i+1]

        # Segmenti che attraversano N_target
        if (N1 - N_target)*(N2 - N_target) <= 0 and N1 != N2:
            t = (N_target - N1) / (N2 - N1)
            M_interp = M1 + t*(M2 - M1)
            Ms_at_N.append(M_interp)

    if not Ms_at_N:
        return 0.0, 0.0

    Ms_at_N = np.array(Ms_at_N)

    M_pos = np.max(Ms_at_N)   # ramo positivo
    M_neg = np.min(Ms_at_N)   # ramo negativo

    return float(M_pos), float(M_neg)
# -----------------------------------------------------------------------------------------------
# ===================== Contorno ellissoidale Mx-My a N = cost (quadranti) =====================
def mxmy_ellipsoid_contour_quadrants(
        N_target,
        N_hull_x, M_hull_x,
        N_hull_y, M_hull_y,
        n_phi=360):

    # Estrai MxRd+ e MxRd−
    Mx_pos, Mx_neg = bending_capacities_at_N(N_hull_x, M_hull_x, N_target)
    My_pos, My_neg = bending_capacities_at_N(N_hull_y, M_hull_y, N_target)

    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    Mx_list = []
    My_list = []

    for phi in phis:
        c = math.cos(phi)
        s = math.sin(phi)

        # Seleziona i valori resistenti in base al quadrante
        MxRd = Mx_pos if c >= 0 else Mx_neg
        MyRd = My_pos if s >= 0 else My_neg

        # Evita divisioni per zero
        if abs(MxRd) < 1e-9 or abs(MyRd) < 1e-9:
            Mx_list.append(0)
            My_list.append(0)
            continue

        # Formula ellissoidale
        denom = (abs(c)**1.5)/(abs(MxRd)**1.5) + (abs(s)**1.5)/(abs(MyRd)**1.5)
        if denom <= 0:
            M = 0
        else:
            M = denom**(-1/1.5)

        Mx_list.append(M * c)
        My_list.append(M * s)

    # Chiudi la curva
    Mx_arr = np.array(Mx_list)
    My_arr = np.array(My_list)
    Mx_closed = np.concatenate([Mx_arr, Mx_arr[:1]])
    My_closed = np.concatenate([My_arr, My_arr[:1]])

    return Mx_closed, My_closed

# ===================== EC2 parabola‑rettangolo: M–N come inviluppo (εm, κ) =====================
def sigma_c_ec2(eps, fcd, eps_c2, eps_cu):
    """
    Parabola‑rettangolo EC2 (solo compressione):
      eps <= 0: 0
      0<eps<=ec2: fcd*(2η-η^2), η=eps/ec2
      ec2<eps<=ecu: fcd
      eps>ecu: fcd (clamp)
    """
    if eps <= 0.0:
        return 0.0
    if eps <= eps_c2:
        eta = eps / eps_c2
        return fcd * (2.0*eta - eta*eta)
    if eps <= eps_cu:
        return fcd
    return fcd

def monotonic_chain_hull(points):
    """
    Invilluppo convesso (hull) 2D di una nuvola di punti (x,y) – O(n log n).
    Ritorna lista di vertici (x,y) in senso CCW (chiusa con ripetizione del primo).
    """
    P = sorted(set((float(x), float(y)) for (x,y) in points))
    if len(P) <= 1:
        return P + P

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if not hull:
        return P + P
    hull.append(hull[0])  # chiudi
    return hull

def MN_envelope_ec2(axis, b, h, rebars, mat,
                    n_eps=41, n_kappa=81, n_fib=240,
                    eps_m_neg_factor=4.0, kappa_sy_factor=6.0, kappa_c_factor=2.0):
    """
    Genera il **cloud** di stati (N,M) con integrazione a fibre per EC2,
    campionando la **deformazione media** εm e la **curvatura** κ:
        eps(y/x) = εm + κ * (y o x)
    - axis='x' -> restituisce inviluppo nel piano (N, Mx)
    - axis='y' -> restituisce inviluppo nel piano (N, My)
    Copre N>0 e N<0 (compressione/trazione) e M ±.
    Restituisce: (N_cloud, M_cloud, N_hull, M_hull)
    """
    fcd = mat['fcd']; eps_cu = mat['eps_cu']; eps_c2 = mat['eps_c2']
    Es = mat['Es']; fyd = mat['fyd']
    eps_sy = fyd / Es

    # Discretizzazione fibre lungo lo spessore (direzione del gradiente)
    if axis == 'x':
        depth = h; width = b
        n_f = max(60, int(n_fib))
        dy = depth / n_f
        coords = np.linspace(-depth/2 + 0.5*dy, depth/2 - 0.5*dy, n_f)
    elif axis == 'y':
        depth = b; width = h
        n_f = max(60, int(n_fib))
        dx = depth / n_f
        coords = np.linspace(-depth/2 + 0.5*dx, depth/2 - 0.5*dx, n_f)
    else:
        raise ValueError("axis deve essere 'x' o 'y'")

    # Range εm: fino a trazione importante dell'acciaio
    eps_m_min = -eps_m_neg_factor * eps_sy
    eps_m_max = +1.20 * eps_cu
    eps_ms = np.linspace(eps_m_min, eps_m_max, int(n_eps))

    # Range κ: raggiungere snervamento acciaio + blocco cls
    k_max = (kappa_sy_factor * eps_sy + kappa_c_factor * eps_cu) / max(depth, 1e-9)
    kappas = np.linspace(-k_max, +k_max, int(n_kappa))

    N_list = []; M_list = []

    for eps_m in eps_ms:
        for kappa in kappas:
            # --- Calcestruzzo (parabola‑rettangolo EC2, solo compressione) ---
            Nc = 0.0; Mc = 0.0
            for s in coords:
                eps = eps_m + kappa * s
                sc = sigma_c_ec2(eps, fcd, eps_c2, eps_cu)  # N/mm^2
                dA = width * (dy if axis=='x' else dx)
                Fc = sc * dA
                Nc += Fc
                if axis == 'x':
                    Mc += s * Fc           # Mx
                else:
                    Mc += -s * Fc          # My (segno coerente con definizione)

            # --- Acciaio (bilineare perf. plastico) ---
            Ns = 0.0; Ms = 0.0
            for (x,y,phi,As) in rebars:
                if axis == 'x':
                    eps_s = eps_m + kappa * y
                    sig_s = max(-fyd, min(fyd, Es * eps_s))
                    Fs = sig_s * As
                    Ns += Fs; Ms += y * Fs
                else:
                    eps_s = eps_m + kappa * x
                    sig_s = max(-fyd, min(fyd, Es * eps_s))
                    Fs = sig_s * As
                    Ns += Fs; Ms += -x * Fs

            N_tot = (Nc + Ns)/1000.0
            M_tot = (Mc + Ms)/1e6
            N_list.append(N_tot); M_list.append(M_tot)

    # Cloud completo e inviluppo convesso (curva chiusa)
    pts = list(zip(N_list, M_list))
    hull = monotonic_chain_hull(pts)
    N_hull = np.array([p[0] for p in hull])
    M_hull = np.array([p[1] for p in hull])

    return np.array(N_list), np.array(M_list), N_hull, M_hull

# ***** funzioni per verifiche numeriche
#------------------------------------------------------------------------------------------------------------------------------
# funzioni per trovare congiungente origine - punto di inviluppo (N_hull, M_hull) e ricavare Momento limite a N_target
# ------------------------------------------------------------------------------------------------------------------------------

# ---------------------------
# ORDINAMENTO E CHIUSURA HULL
# ---------------------------
def sort_and_close_hull(N, M):
    pts = np.column_stack((N, M))

    # Ordina in senso antiorario
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    order = np.argsort(angles)
    pts_sorted = pts[order]

    # Chiudi la polilinea
    if not np.allclose(pts_sorted[0], pts_sorted[-1]):
        pts_sorted = np.vstack([pts_sorted, pts_sorted[0]])

    return pts_sorted[:,0], pts_sorted[:,1]

# -----------------------------------------
# intersezione tra retta OP e segmento AB
# -----------------------------------------
def line_segment_intersection(O, P, A, B):
    O = np.array(O)
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    d = P - O
    v = B - A

    M = np.column_stack((d, -v))
    rhs = A - O

    if abs(np.linalg.det(M)) < 1e-12:
        return None

    t, s = np.linalg.solve(M, rhs)

    # s ∈ [0,1] → intersezione sul segmento
    if s < 0 or s > 1:
        return None

    X = O + t*d
    return X, t

# ------------------------------------
# intersezione retta OP con inviluppo
# ------------------------------------
def intersection_with_hull_from_origin(N, M, N_hull, M_hull):       # vale anche per il contorno Mx,My a N = cost
    O = (0.0, 0.0)
    P = (N, M)

    best_t = None
    best_point = None

    for i in range(len(N_hull)-1):
        A = (N_hull[i],   M_hull[i])
        B = (N_hull[i+1], M_hull[i+1])

        res = line_segment_intersection(O, P, A, B)
        if res is not None:
            X, t = res
            if t >= 0:  # direzione da O verso P
                if best_t is None or t < best_t:
                    best_t = t
                    best_point = X

    return best_point, best_t


# ------------------------------------------
# COEFFICIENTE DI SICUREZZA DIAG. N,M o MX,MY
# ------------------------------------------
def safety_factor_origin(Px, Py, Px_contour, Py_contour):
    """
    Fattore di sicurezza nel piano Mx–My:
    γ = |P - O| / |P_lim - O|
    """
    P_lim, t = intersection_with_hull_from_origin(Px, Py, Px_contour, Py_contour)  # utilizzo la stessa function di N,M N_hull, M_hull perché è generica per qualsiasi contorno chiuso
    if P_lim is None:
        return None, None

    P = np.array([Px, Py], dtype=float)
    O = np.array([0.0, 0.0], dtype=float)
    P_lim = np.array(P_lim, dtype=float)

    num = np.linalg.norm(P - O)
    den = np.linalg.norm(P_lim - O)

    if den < 1e-12:
        return None, None

    gamma = num / den
    return gamma, P_lim

# --------------------------------------------------------------------------------------
# Funzioni per trovare limiti nei grafici
# --------------------------------------------------------------------------------------
def floor_to_step(value, step):
    return step * int(value // step)

def ceil_to_step(value, step):
    return step * int(-(-value // step))  # ceiling con divisione intera



# --------------------------------------------------------------------------------------------------
# ============== ***  INTERFACCIA UTENTE * UI Streamlit *** ====================================
# --------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Domain N-Mx-My | rc.",
    #layout="wide"
    )
#st.title("Dominio N-Mx-My e diagrammi M-N (curve chiuse) — sezione in c.a.")
st.title("Interaction diagrams visualization")

with st.expander("🆘 Details & limits of the model", expanded=False):
    st.markdown("""
- **3D Domain**: uniform block `σc = ηc·fcd` (concrete resistant only to compression), steel bilinear ±`fyd`. θ in the range **[0, 2π)**.
- **Mx-N** e **My-N**: **envelope** in (N,M) obtained from **grid 2D (εm, κ)** by **fiber** integration with law **EC2 parabola-rectangle** for concrete.
  Steel is elastic-perfectly plastic (±`fyd`). Covered **N>0** and **N<0**.
- **Axes** of M-N planes: **N** → abscissas, **M** → ordinates.
- **Use for Design**: the tool is compliant with EC2/NTC code.
""")

# --------------------------------------------------------------------
# Lettura dati dinamici dalla sessione ID utente e statici da files
# --------------------------------------------------------------------
sessionDir = "sessions"
fileConcrete = os.path.join(sessionDir, f"concrete_{st.session_state.session_id}.csv")
fileSteel = os.path.join(sessionDir, f"steel_{st.session_state.session_id}.csv")
fileDatiGeo = f"{sessionDir}/DatiGeo_{st.session_state.session_id}.csv"
fileReinfLayers = os.path.join(sessionDir, f"Reinf_layers_{st.session_state.session_id}.csv")

fileDatiLoads = os.path.join(sessionDir, f"DatiLoad_{st.session_state.session_id}.csv")


if not os.path.exists(fileConcrete) or not os.path.exists(fileSteel) or not os.path.exists(fileDatiGeo) or not os.path.exists(fileReinfLayers):
    st.warning("Missing Data for this session. Please ensure all previous steps are completed.")
    st.stop()

df_concrete = pd.read_csv(fileConcrete, sep=',')
df_steel = pd.read_csv(fileSteel, sep=',')
df_geo = pd.read_csv(fileDatiGeo)

df_loads = pd.read_csv(fileDatiLoads, sep=',') if os.path.exists(fileDatiLoads) else pd.DataFrame()
# Rinomina eventuali colonne con spazi o maiuscole df.columns = df.columns.str.strip()

if df_loads.columns.size > 0:
    df_loads.columns = df_loads.columns.str.strip()

#print(df_geo.columns)
#print(df_geo.head())

st.session_state.fcd = df_concrete.loc[0,'fcd']
st.session_state.fyd = df_steel.loc[0,'fyd']
st.session_state.Es = df_steel.loc[0,'Es']
st.session_state.base = df_geo.loc[0,'base']
st.session_state.height = df_geo.loc[0,'height']
st.session_state.cover = df_geo.loc[0,'cover']
st.session_state.stirrups_dia = df_geo.loc[0,'stirrups_dia']

fcd = st.session_state.fcd
fyd = st.session_state.fyd 
Es = st.session_state.Es 
b = st.session_state.base
h = st.session_state.height
c = st.session_state.cover
stirrups_dia = st.session_state.stirrups_dia    

eta_c = 0.80        # Blocco rettangolare 3D
eps_cu = 0.0035     # allungamento ultimo cls
eps_c2 = 0.0020     # allungamento inizio plateau cls

#("Dominio 3D (rapido)")
n_theta = 72
n_c = 60
show_mesh = True    # mostra mesh 3D
#("Taglio Mx-My a N = cost.")
N_target = 0.0      # Sforzo normale kN
tolN = 2.0          # Tolleranza N kN
#("M-N (EC2, inviluppo chiuso)")
n_eps = 41          # Punti εm
n_kappa = 81        # punti κ
n_fib = 240         # Fibre nello spessore
show_cloud = False  # mostra nuvola di punti



# Preparazione sezione e armature
# rebars_from_layers(csv_path, section_height) --> ritorna lista di tuple (x, y, phi, As) in rebars_pts
poly = rect_section(b, h)
#rebar_pts, As_bar = edge_rebars(b, h, cover, phi, nx_top, nx_bot, ny_left, ny_right)
rebar_pts = rebars_from_layers(fileReinfLayers, b, h)

#print(rebar_pts)

# Materiali
mat = {'fcd': fcd, 'eta_c': eta_c, 'fyd': fyd, 'Es': Es, 'eps_cu': eps_cu, 'eps_c2': eps_c2}

# --- N_ultimo di compressione secondo formula: N_ult = fyd*As_tot + fcd*Ac/1.25 ---
As_tot = sum(As for (_, _, _, As) in rebar_pts)     # mm^2
#As_tot = len(rebar_pts) * As_bar         # mm^2
Ac = b * h                               # mm^2 (sezione piena rettangolare)
N_ult_comp = (mat['fyd'] * As_tot + mat['fcd'] * Ac * 0.8) / 1000.0   # kN


# --------------------------------------------------------------------------------------------------
# ===================== *** Dominio 3D *** =====================
# --------------------------------------------------------------------------------------------------
col1, col2 = st.columns([2, 1], gap="large")

if st.checkbox("Show 3D interaction domain", value=False):
    Domain3D = True  

    st.subheader("3D Interaction Domain N-Mx-My (θ∈[0,2π))")
    with st.spinner("Domain calculation..."):
        thetas, cs_grid, Ns_g, Mxs_g, Mys_g, Ls = sample_domain_grid(
            poly, rebar_pts, mat, n_theta=n_theta, n_c=n_c
        )
    Ns_flat = Ns_g.flatten(); Mxs_flat = Mxs_g.flatten(); Mys_flat = Mys_g.flatten()
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=Mxs_flat, y=Mys_flat, z=Ns_flat,
        mode='markers', marker=dict(size=2.8, color=Ns_flat, colorscale='Turbo', opacity=0.85),
        name='Campionamento'
    ))
    if show_mesh:
        I, J, K = grid_to_mesh_indices(n_theta, n_c, wrap_theta=True)
        fig3d.add_trace(go.Mesh3d(
            x=Mxs_flat, y=Mys_flat, z=Ns_flat,
            i=I, j=J, k=K, color='lightsteelblue', opacity=0.35, name='Mesh'
        ))
    fig3d.update_layout(
        scene=dict(xaxis_title="Mx [kNm]", yaxis_title="My [kNm]", zaxis_title="N [kN]"),
        height=660
    )
    st.plotly_chart(fig3d, use_container_width=True)
else:
    Domain3D = False

# ------------------------------------------------------------------------------------------------
# ===================== Diagrammi Mx–N e My–N (EC2, inviluppi chiusi) =====================
# ------------------------------------------------------------------------------------------------

sf_results = {}     # dizionario dove caricheremo i risultati dei coefficienti di sicurezza per ogni combinazione di carico

# -----------------------------------------------
# ===  Mx-N  ===
# -----------------------------------------------
with st.spinner("EC2 fiber integration on grid (εm, κ)…"):
    # Mx–N
    N_cloud_x, M_cloud_x, N_hull_x, M_hull_x = MN_envelope_ec2(
        'x', b, h, rebar_pts, mat,
        n_eps=n_eps, n_kappa=n_kappa, n_fib=n_fib
    )

    # My–N
    N_cloud_y, M_cloud_y, N_hull_y, M_hull_y = MN_envelope_ec2(
        'y', b, h, rebar_pts, mat,
        n_eps=n_eps, n_kappa=n_kappa, n_fib=n_fib
    )

# 🔥 ORDINAMENTO + CHIUSURA (OBBLIGATORIO)
N_hull_x, M_hull_x = sort_and_close_hull(N_hull_x, M_hull_x)
N_hull_y, M_hull_y = sort_and_close_hull(N_hull_y, M_hull_y)

# Mx–N
N_min_x = min(N_hull_x)
N_max_x = max(N_hull_x)
M_min_x = min(M_hull_x)
M_max_x = max(M_hull_x)
tick_Nxstep = 200
tick_Nxmin = floor_to_step(N_min_x, tick_Nxstep)
tick_Nxmax = ceil_to_step(N_max_x, tick_Nxstep)
tick_Mx_step = 50
tick_Mx_min = floor_to_step(M_min_x, tick_Mx_step)
tick_Mx_max = ceil_to_step(M_max_x, tick_Mx_step)

# My–N
N_min_y = min(N_hull_y)
N_max_y = max(N_hull_y)
M_min_y = min(M_hull_y)
M_max_y = max(M_hull_y)
tick_Nystep = 200
tick_Nymin = floor_to_step(N_min_y, tick_Nystep)
tick_Nymax = ceil_to_step(N_max_y, tick_Nystep)
tick_My_step = 50
tick_My_min = floor_to_step(M_min_y, tick_My_step)
tick_My_max = ceil_to_step(M_max_y, tick_My_step)


if st.checkbox("Show diagrams Mx-N and My-N  (Envelopes)", value=True):
    st.subheader("Diagrams Mx-N & My-N")
    diagMN = True    
    
    # Plot Mx–N
    fig_mx = go.Figure()
    if show_cloud:
        fig_mx.add_trace(go.Scatter(x=N_cloud_x, y=M_cloud_x, mode='markers',
                                    marker=dict(size=3, color='lightgray'), name='Cloud'))
    fig_mx.add_trace(go.Scatter(x=N_hull_x, y=M_hull_x, mode='lines', name='Envelope Mx-N',
                                line=dict(color='royalblue', width=2)))

    if not df_loads.empty:
        marker_symbol = 'diamond'
        fig_mx.add_trace(
            go.Scatter(
                x=df_loads["axial"],
                y=df_loads["momentX"],
                mode="markers+text",
                marker=dict(size=7, color="red", symbol=marker_symbol),
                text=df_loads["Load"],
                textposition="top center",
                textfont=dict(size=9, color="darkred"),
                name="Load combination"
            )
        )

    fig_mx.update_layout(xaxis_title="N [kN] (compression +)", yaxis_title="Mx [kNm]", height=430)


    # -------------- Calcolo e visualizzazione coefficienti di sicurezza --------------------------
    
    for idx, row in df_loads.iterrows():
        N = row["axial"]
        M = row["momentX"]

        gamma, P_lim = safety_factor_origin(N, M, N_hull_x, M_hull_x)
        
        if gamma is not None:
            fig_mx.add_trace(go.Scatter(
                x=[0, P_lim[0]],
                y=[0, P_lim[1]],
                mode="lines",
                line=dict(color="orange", width=1, dash="dot"),
                showlegend=False
            ))

            fig_mx.add_trace(go.Scatter(
                x=[P_lim[0]],
                y=[P_lim[1]],
                mode="markers",
                marker=dict(size=8, color="orange", symbol="circle-open"),
                #name=f"Limit point {row['Load']}"
                showlegend=False
            ))
            col1, col2 = st.columns(2)
            col1.write(f"Load **{row['Load']}** → γ = **{gamma:.3f}**")

            if gamma > 1.0:
                if  M > tick_Mx_min and M < tick_Mx_max and N > tick_Nxmin and N < tick_Nxmax:               
                    col2.write("⚠️ Load lies outside the interaction domain boundary!")
            
                else:
                
                    col2.write("⚠️ Load lies outside the interaction domain boundary and outside plot limits!")
 
            load_name = row["Load"]
            if load_name not in sf_results:
                sf_results[load_name] = {"SFx": None, "SFy": None, "SFm": None}

            sf_results[load_name]["SFx"] = gamma

        else:
            st.write(f"Load **{row['Load']}** → no intersection found (unexpected)")
   
    # ------fine coefficienti di sicurezza ---------------------------------------------------------------------------

    # --- Linea verticale su Mx–N passante per x=0 --- 
    grid_color = "rgba(200,200,200,0.2)"
    fig_mx.add_vline(
        x=0.0,
        line=dict(color= "grey", width= 1, dash="solid"),
        annotation_text="",
        annotation_position="top right"
    )
    # --- Linea orizzontale su Mx–N passante per y=0 ---
    fig_mx.add_hline(
        y=0.0,
        line=dict(color= "grey", width= 1, dash="solid"),
        annotation_text="",
        annotation_position="top right"
    )

    fig_mx.add_vline(
        x=N_ult_comp,
        line=dict(color="firebrick", width=1.5, dash="dash"),
        annotation_text=f"Nu ≈ {N_ult_comp:.0f} kN",
        annotation_position="top right"
    )


    # Attiva griglia verticale (asse x = N)

    fig_mx.update_xaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_Nxstep, range=[tick_Nxmin, tick_Nxmax])
    fig_mx.update_yaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_Mx_step, range=[tick_Mx_min, tick_Mx_max])


    # Salvataggio su file PNG
    fileImgMx = os.path.join(sessionDir, f"imgMx_{st.session_state.session_id}.png")
    fig_mx.write_image(fileImgMx, scale=3)

    st.plotly_chart(fig_mx, use_container_width=True)

    # -------------------------------------------------------------
    # *** Plot My–N ***
    # ------------------------------------------------------------
    fig_my = go.Figure()
    if show_cloud:
        fig_my.add_trace(go.Scatter(x=N_cloud_y, y=M_cloud_y, mode='markers',
                                    marker=dict(size=3, color='lightgray'), name='Cloud'))
    fig_my.add_trace(go.Scatter(x=N_hull_y, y=M_hull_y, mode='lines', name='Envelope My-N',
                                line=dict(color='seagreen', width=2)))

    if not df_loads.empty:
        marker_symbol = 'diamond'
        fig_my.add_trace(
            go.Scatter(
                x=df_loads["axial"],
                y=df_loads["momentY"],
                mode="markers+text",
                marker=dict(size=7, color="red", symbol=marker_symbol),
                text=df_loads["Load"],
                textposition="top center",
                textfont=dict(size=9, color="darkred"),
                name="Load combination"
            )
        )


    fig_my.update_layout(xaxis_title="N [kN] (compression +)", yaxis_title="My [kNm]", height=430)

    # -------------- Calcolo e visualizzazione coefficienti di sicurezza --------------------------


    col1, col2= st.columns(2)

    for idx, row in df_loads.iterrows():
        N = row["axial"]
        M = row["momentY"]

        gamma_y, P_lim_y = safety_factor_origin(N, M, N_hull_y, M_hull_y)

        if gamma_y is not None:
            fig_my.add_trace(go.Scatter(
                x=[0, P_lim_y[0]],
                y=[0, P_lim_y[1]],
                mode="lines",
                #line=dict(color="orange", width=1, dash="dot"),
                line=dict(color="orange", width=1, dash="dot"),
                showlegend=False
            ))


            fig_my.add_trace(go.Scatter(
                x=[P_lim_y[0]],
                y=[P_lim_y[1]],
                mode="markers",
                marker=dict(size=8, color="orange", symbol="circle-open"),
                #name=f"Limit point {row['Load']}"
                #text=f"Limit point for {row['Load']}",
                #hoverinfo="text",
                showlegend=False
            ))

            col1.write(f"Load **{row['Load']}** → γ = **{gamma_y:.3f}**")

            if gamma_y > 1.0:
                if  M > tick_My_min and M < tick_My_max and N > tick_Nymin and N < tick_Nymax:               
                    col2.write("⚠️ Load lies outside the interaction domain boundary!")           
                else:
                    col2.write("⚠️ Load lies outside the interaction domain boundary and outside plot limits!")
            


            load_name = row["Load"]
            if load_name not in sf_results:
                sf_results[load_name] = {"SFx": None, "SFy": None, "SFm": None}

            sf_results[load_name]["SFy"] = gamma

        else:
            st.write(f"Load **{row['Load']}** → no intersection found (unexpected)")

    # ------fine -------------------------------------------------------------------------------

    # --- Linea verticale su Mx–N passante per x=0 --- 
    grid_color = "rgba(200,200,200,0.2)"
    fig_my.add_vline(
        x=0.0,
        line=dict(color= "grey", width= 1, dash="solid"),
        annotation_text="",
        annotation_position="top right"
    )
    # --- Linea orizzontale su Mx–N passante per y=0 ---
    fig_my.add_hline(
        y=0.0,
        line=dict(color= "grey", width= 1, dash="solid"),
        annotation_text="",
        annotation_position="top right"
    )
        
    fig_my.add_vline(
        x=N_ult_comp,
        line=dict(color="firebrick", width=1.5, dash="dash"),
        annotation_text=f"Nu ≈ {N_ult_comp:.0f} kN",
        annotation_position="top right"
    )

    fig_my.update_xaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_Nystep, range=[tick_Nymin, tick_Nymax])
    fig_my.update_yaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_My_step, range=[tick_My_min, tick_My_max])
    #fig_my.update_xaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_Nystep)
    #fig_my.update_yaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, dtick=tick_My_step)


    # Salvataggio su file PNG
    fileImgMy = os.path.join(sessionDir, f"imgMy_{st.session_state.session_id}.png")
    fig_my.write_image(fileImgMy, scale=3)

    st.plotly_chart(fig_my, use_container_width=True)
else:
    diagMN = False    


# ----------------------------------------------------------------------------------------------------
# ===================== Domini (Mx–My) per i valori di N presenti nei carichi =====================
# ----------------------------------------------------------------------------------------------------

if st.checkbox("Show domain Mx-My for all N in df_loads", value=True):
    CurveMxMy = True

    st.subheader("Mx-My Domains for all N in df_loads")

    if df_loads.empty:
        st.warning("Load combinations not loaded.")
        all_rows = []
    else:
        # Estrai tutti i valori unici di N_target dal CSV
        N_values = df_loads["axial"].unique()
        N_values = np.sort(N_values)

        st.write(f"Found {len(N_values)} distinct N values in df_loads.")

        fig2d = go.Figure()

        # --- Colori casuali per ogni curva ---
        import random
        import colorsys


        def random_color():
            h = random.random()
            s = 0.6 + random.random()*0.3
            v = 0.7 + random.random()*0.3
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
            #    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


        # --- Traccia tutte le curve Mx–My per ogni N_target ---
        with st.spinner("Computing all Mx–My slices..."):
            all_rows = []
            for N_target in N_values:
                Mx_closed, My_closed = mxmy_ellipsoid_contour_quadrants(
                    N_target,
                    N_hull_x, M_hull_x,
                    N_hull_y, M_hull_y,
                    n_phi=360
                )
                # Ordina e chiudi il contorno Mx–My per questo N_target
                Mx_closed, My_closed = sort_and_close_hull(Mx_closed, My_closed)  # riga aggiunta per ordinamento e chiusura. Usa la stessa function già definita per N,M perché è generica per qualsiasi contorno chiuso

                # 1. curva Mx–My
                color = random_color()
                fig2d.add_trace(
                    go.Scatter(
                        x=Mx_closed,
                        y=My_closed,
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f"N = {N_target:.0f} kN"
                    )
                )

                # 2. FILTRA i punti relativi a quel N_target
                df_sliceN = df_loads[df_loads["axial"] == N_target]

                # --- Aggiungi solo i punti Mx–My del CSV relativi a N_target---
                fig2d.add_trace(
                    go.Scatter(
                        x=df_sliceN["momentX"],
                        y=df_sliceN["momentY"],
                        mode="markers+text",
                        marker=dict(size=9, color=color, symbol="diamond"),
                        text=df_sliceN["Load"],
                        textposition="top center",
                        textfont=dict(size=9, color="darkred"),
                        #name="Load Combination"
                        name= f"Loads: {df_sliceN['Load'].iloc[0]}" # <-- nome della trace = prima riga
                    )
                )
                
                
                # --- Calcolo fattore di sicurezza per ogni punto Mx–My di questo N_target ---
                for _, row in df_sliceN.iterrows():
                    Mx_load = row["momentX"]
                    My_load = row["momentY"]

                    gamma, P_lim = safety_factor_origin(Mx_load, My_load, Mx_closed, My_closed)

                    if gamma is not None:
                        # Congiungente (0,0) -> punto di intersezione sul contorno
                        fig2d.add_trace(
                            go.Scatter(
                                x=[0.0, P_lim[0]],
                                y=[0.0, P_lim[1]],
                                mode="lines",
                                line=dict(color=color, width=1, dash="dot"),
                                showlegend=False
                            )
                        )

                        # Punto di intersezione sul contorno
                        fig2d.add_trace(
                            go.Scatter(
                                x=[P_lim[0]],
                                y=[P_lim[1]],
                                mode="markers",
                                marker=dict(size=7, color=color, symbol="circle-open"),
                                showlegend=False
                            )
                        )

                        col1, col2 = st.columns(2)
                        
                        col1.write(
                            f"N = {N_target:.0f} kN, Load **{row['Load']}** → γ = {gamma:.3f}"
                        )                        
                        
                        if gamma > 1.0:
                            if  Mx_load > tick_Mx_min and Mx_load < tick_Mx_max and My_load > tick_My_min and My_load < tick_My_max:               
                                col2.write("⚠️ Load lies outside the interaction domain boundary!")           
                            else:
                                col2.write("⚠️ Load lies outside the interaction domain boundary and outside plot limits!")
                            #col2.markdown(
                            #    "<div style='background-color:#ffe6e6; color:red; padding:6px; "
                            #    "border-radius:4px; font-weight:bold;'>⚠️ Load outside the domain!</div>",
                            #    unsafe_allow_html=True
                            #)


                        load_name = row["Load"]
                        if load_name not in sf_results:
                            sf_results[load_name] = {"SFx": None, "SFy": None, "SFm": None}

                        sf_results[load_name]["SFm"] = gamma

                    else:
                        st.write(
                            f"N = {N_target:.0f} kN, Load **{row['Load']}** → no intersection found (check contour)"
                        )
                # --------------------Fine routine coefficiente di sicurezza--------------------    

                for mx, my in zip(Mx_closed, My_closed):
                        all_rows.append({
                            "N_target": N_target,
                            "Mx": mx,
                            "My": my
                        })


        # --- Layout ---
        fig2d.update_layout(
            xaxis_title="Mx [kNm]",
            yaxis_title="My [kNm]",
            width=520,
            height=520,
            legend=dict(font=dict(size=10))
        )

        fig2d.update_yaxes(scaleanchor="x", scaleratio=1)

        df_all = pd.DataFrame(all_rows)
        Mx_min = df_all["Mx"].min()
        Mx_max = df_all["Mx"].max()
        My_min = df_all["My"].min()
        My_max = df_all["My"].max()


        step = 50
        Mx_min_tick = floor_to_step(Mx_min, step)
        Mx_max_tick = ceil_to_step(Mx_max, step)
        My_min_tick = floor_to_step(My_min, step)
        My_max_tick = ceil_to_step(My_max, step)

        #print("Tick Mx range:", Mx_min_tick, "to", Mx_max_tick)
        #print("Tick My range:", My_min_tick, "to", My_max_tick)

        # --- Linea verticale su Mx–My passante per x=0 --- 
        grid_color = "rgba(200,200,200,0.2)"
        fig2d.add_vline(
            x=0.0,
            line=dict(color= "grey", width= 1, dash="solid"),
            annotation_text="",
            annotation_position="top right"
        )
        # --- Linea orizzontale su Mx–My passante per y=0 ---
        fig2d.add_hline(
            y=0.0,
            line=dict(color= "grey", width= 1, dash="solid"),
            annotation_text="",
            annotation_position="top right"
        )


        grid_color = "rgba(200,200,200,0.2)"
        fig2d.update_xaxes(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            dtick=step,
            range=[Mx_min_tick, Mx_max_tick],
            autorange=False,
            constrain='domain'
        )
        fig2d.update_yaxes(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            dtick=step,
            range=[My_min_tick, My_max_tick],
            autorange=False,
            constrain='domain'
        )   

        fig2d.update_layout( xaxis=dict(range=[Mx_min_tick, Mx_max_tick], autorange=False), 
        yaxis=dict(range=[My_min_tick, My_max_tick], autorange=False)
        )

        # Salvataggio su file PNG
        fileImgMxMy = os.path.join(sessionDir, f"imgMxMy_{st.session_state.session_id}.png")
        fig2d.write_image(fileImgMxMy, scale=3)
        
        st.plotly_chart(fig2d, use_container_width=True)

else:
    CurveMxMy = False

# ===================== Tabella riepilogativa coefficienti di sicurezza con valutazione =====================
rows = []
for load_name, vals in sf_results.items():
    SFx = vals["SFx"]
    SFy = vals["SFy"]
    SFm = vals["SFm"]

    # Check finale
    if SFx is not None and SFy is not None and SFm is not None:
        chk = "Ok" if (SFx <= 1 and SFy <= 1 and SFm <= 1) else "Fail"
    else:
        chk = "Incomplete"

    rows.append({
        "Load_C": load_name,
        "SF(Mx-N)": f"{SFx:.2f}",
        "SF(My-N)": f"{SFy:.2f}",
        "SF(Mx-My)": f"{SFm:.2f}",
        "Check": chk
    })

df_sf = pd.DataFrame(rows)

fileSF = os.path.join(sessionDir, f"SafetyFactors_{st.session_state.session_id}.csv")
df_sf.to_csv(fileSF, index=False)

# st.success(f"Safety factor table saved to: {fileSF}")
def style_cells(v):
    num = pd.to_numeric(v, errors="coerce")

    # Safety factors
    if num > 1.0:
        return "background-color: red; color: white; font-weight: bold;"
    if 0.9 <= num <= 1.0:
        return "background-color: yellow; color: black; font-weight: bold;"

    # Check column
    if v == "Ok":
        return "background-color: green; color: white; font-weight: bold;"
    if v == "Fail":
        return "background-color: red; color: white; font-weight: bold;"

    return ""

styled_df = df_sf.style.applymap(style_cells)

st.markdown("### Checks Summary with Safety Factors")
st.dataframe(styled_df, hide_index=True)



# ===================== Export CSV =====================
if Domain3D or CurveMxMy or diagMN:

    st.subheader("Export data (CSV)")
    # Dominio 3D
    if Domain3D:
        df_dom = pd.DataFrame({'N_kN': Ns_flat, 'Mx_kNm': Mxs_flat, 'My_kNm': Mys_flat})
        buffer = io.BytesIO()
        df_dom.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="3D-Domain -> Excel",
            data=buffer,
            file_name="3D_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


        
        #st.download_button("3D-Domain (CSV)", data=df_dom.to_csv(index=False).encode('utf-8'),
        #                file_name="dominio_N_Mx_My.csv", mime="text/csv")


    # Inviluppi chiusi Mx–N e My–N (N in x, M in y)
    if diagMN:
        df_mx_env = pd.DataFrame({'N_kN': N_hull_x, 'Mx_kNm': M_hull_x})
        df_my_env = pd.DataFrame({'N_kN': N_hull_y, 'My_kNm': M_hull_y})
        buffer1 = io.BytesIO()
        buffer2 = io.BytesIO()
        df_mx_env.to_excel(buffer1, index=False)
        df_my_env.to_excel(buffer2, index=False)
        buffer1.seek(0)
        buffer2.seek(0) 

        #st.download_button("Inviluppo Mx–N (CSV)", data=df_mx_env.to_csv(index=False).encode('utf-8'),
        #                file_name="invilluppo_Mx_N.csv", mime="text/csv")
        #st.download_button("Inviluppo My–N (CSV)", data=df_my_env.to_csv(index=False).encode('utf-8'),
        #                file_name="invilluppo_My_N.csv", mime="text/csv")

        st.download_button(
            label="Envelope Mx-N -> Excel",
            data=buffer1,
            file_name="Mx_N_Envelope.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            label="Envelope My-N -> Excel",
            data=buffer2,
            file_name="My_N_Envelope.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    # Contorno Mx–My a N cost
    if CurveMxMy:
        df_all = pd.DataFrame(all_rows)
        buffer_all = io.BytesIO()
        df_all.to_excel(buffer_all, index=False)
        buffer_all.seek(0)
        st.download_button(f"Contours Mx-My -> Excel",
                        data=buffer_all,
                        file_name=f"AllContours_MxMy.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Contorno Mx–My a N cost
    #if SliceMxMy:
    #    df_slice = pd.DataFrame({'Mx_kNm': Mx_closed, 'My_kNm': My_closed})
    #    st.download_button(f"Contour Mx–My a N={int(N_target)} kN (CSV)",
    #                    data=df_slice.to_csv(index=False).encode('utf-8'),
    #                    file_name=f"Contour_MxMy_N_{int(N_target)}kN.csv", mime="text/csv")



st.markdown("---")
st.success("Diagrams Ready!")
