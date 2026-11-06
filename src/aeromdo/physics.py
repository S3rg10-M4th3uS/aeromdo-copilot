"""Python/NumPy port of the TRESIF AeroDesign MDO Simulator (Aerodynamics, Stability, Structures/PV, and Takeoff).

Implements:
- Surface geometry & Munk's Equivalent Monoplane Theorem (`GEOMETRIA`, `MDO__MONOPLAS_EQ.m`)
- Lifting-Line Theory in free flight & ground effect (`LIFT_LINE.m`, `LIFT_LINE_SOLO.m`)
- Parasite drag buildup (`CD0_COMPLETO.m` - Sadraey Ch. 3)
- Nonlinear wing-tail downwash (`DOWNWASH.m`)
- Longitudinal static stability (`AV_ESTAB.m`)
- Component buildup Empty Weight (`PV_AVIAO.m`, `PV_FUSELAGEM.m`, `PV_NERVURAS.m`, `cascao_*.m`)
- Time-domain takeoff & rotation integration with bisection MTOW search (`MDO__CORRIDA_AOC_OFICIAL.m`)
- Flight score `PVOO = 50 * EE * f(N) * 1.15^-b` (Eq. 1.1, Relatório 2026) and Quadratic Penalty (`MDO__FUNCAO_OBJETIVO.m`)
- Design of Experiments via Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`, Sec. 4.2, Relatório 2026)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc

from aeromdo.schemas import (
    AircraftInput,
    SimulationConstraints,
    SimulationOutputs,
    SimulationResult,
)


@dataclass
class AirfoilProfile:
    """Airfoil aerodynamic parameters from BANCO_PERFIL_pLLT.m & Table 5.1."""

    name: str
    cl_max: float
    cd_min: float
    t_c: float
    cl_a_rad: float
    aoa_zl_rad: float
    cm: float


def get_airfoil_profile(profile_id: int) -> AirfoilProfile:
    """Return reference airfoil profile (1 = S1223 wing, 2 = NACA0012 tail)."""
    if profile_id == 1:
        return AirfoilProfile(
            name="S1223",
            cl_max=2.2 * 0.80,
            cd_min=0.015,
            t_c=0.1214,
            cl_a_rad=((2.153 - 1.169) / 10.0) * (180.0 / math.pi),
            aoa_zl_rad=-11.88 * math.pi / 180.0,
            cm=-0.360,
        )
    return AirfoilProfile(
        name="NACA0012",
        cl_max=1.193,
        cd_min=0.006,
        t_c=0.1200,
        cl_a_rad=(1.044 / 10.0) * (180.0 / math.pi),
        aoa_zl_rad=0.0,
        cm=0.0,
    )


@dataclass
class SurfaceAeroState:
    """Polynomial curves and stall parameters for a lifting surface (flight or ground)."""

    curva_cl: np.ndarray  # [slope, intercept] in deg
    curva_cm: np.ndarray  # [slope, intercept] in deg
    curva_cd: dict[int, np.ndarray]  # velocity -> [a2, a1, a0] in deg
    k_induced: float
    alfa_estol: float
    cl_0: float = 0.0
    cl_a: float = 0.0
    cm_ca: float = 0.0
    cl_max: float = 0.0
    e0: float = 0.0
    de: float = 0.0
    alfa_max: float = 0.0


@dataclass
class LiftingSurface:
    """Geometric and aerodynamic representation of Wing (ASA) or Horizontal Tail (SH)."""

    name: str
    b: float
    cr: float
    cp: float
    pr: float
    inc: float
    pos_ba: np.ndarray
    pos: np.ndarray
    perfil: AirfoilProfile
    s_reta: float = 0.0
    s_afil: float = 0.0
    s: float = 0.0
    ar: float = 0.0
    afil: float = 0.0
    mac: float = 0.0
    lht: float = 0.0
    n_eff: float = 0.9
    gap: float = 0.0
    voo: SurfaceAeroState | None = None
    solo: SurfaceAeroState | None = None


def compute_geometry(sup: LiftingSurface) -> LiftingSurface:
    """Compute area, aspect ratio, taper ratio, and MAC (GEOMETRIA in LLT_ASA.m)."""
    sup.afil = sup.cp / sup.cr if sup.cr > 0 else 1.0
    sup.s_reta = sup.pr * sup.b * sup.cr
    sup.s_afil = (1.0 - sup.pr) * ((sup.cr + sup.cp) / 2.0) * sup.b
    sup.s = sup.s_reta + sup.s_afil
    sup.ar = (sup.b**2) / sup.s if sup.s > 0 else 0.0
    tr = sup.afil
    if sup.s > 0:
        mac_afil = (2.0 / 3.0) * sup.cr * (1.0 + tr + tr**2) / (1.0 + tr)
        sup.mac = (sup.s_afil * mac_afil + sup.s_reta * sup.cr) / sup.s
    else:
        sup.mac = 0.0
    return sup


def apply_equivalent_monoplane(sup: LiftingSurface, gap: float) -> LiftingSurface:
    """Munk's Equivalent Monoplane Theorem for biplane configurations (MDO__MONOPLAS_EQ.m)."""
    b_avg = sup.b
    h_b = np.clip(gap / max(b_avg, 1e-6), 0.05, 0.50)
    mu = 1.0  # Equal-span biplane wings
    sigma_diehl = (6.0 / 75.0) * (((75.0 * mu) - (28.0 + 20.0 * mu) * h_b) / (6.0 + (29.0 * mu - 5.0) * h_b))
    k_munk = math.sqrt(2.0 / (1.0 + sigma_diehl))

    eq = LiftingSurface(
        name=sup.name,
        b=k_munk * b_avg,
        cr=sup.cr,
        cp=sup.cp,
        pr=sup.pr,
        inc=sup.inc,
        pos_ba=sup.pos_ba.copy(),
        pos=sup.pos.copy(),
        perfil=sup.perfil,
        gap=gap,
    )
    # Match total area of the two wings while keeping PR and taper ratio
    total_s = 2.0 * sup.s
    denom = eq.b * (eq.pr + (1.0 - eq.pr) * (1.0 + sup.afil) / 2.0)
    eq.cr = total_s / max(denom, 1e-6)
    eq.cp = eq.cr * sup.afil
    return compute_geometry(eq)


def cd0_completo(sup: LiftingSurface, v: float) -> float:
    """Parasite drag buildup (CD0_COMPLETO.m - Sadraey Cap. 3)."""
    mach = 0.04
    rho = 1.1
    mu = 1.461e-5
    re = max((rho * max(v, 1.0) * max(sup.mac, 0.05)) / mu, 1e3)
    cf = 0.455 / ((math.log10(re)) ** 2.58)
    tc = sup.perfil.t_c
    f_tc = 1.0 + 2.7 * tc + 100.0 * (tc**4)
    f_m = 1.0 - 0.08 * (mach**1.45)
    s_wet_ratio = 2.0 * (1.0 + tc / 2.0)
    return cf * f_tc * f_m * s_wet_ratio * ((sup.perfil.cd_min / 0.004) ** 0.4)


def _solve_lifting_line_coeffs(sup: LiftingSurface) -> tuple[float, float]:
    """Solve N=21 Fourier coefficients in LIFT_LINE.m and return (CL_at_15deg, delta_induced)."""
    n_terms = 21
    cr = sup.cr
    afil = sup.cp / sup.cr
    b = sup.b
    cla = sup.perfil.cl_a_rad
    aoa_zl = sup.perfil.aoa_zl_rad
    aoa = 15.0 * math.pi / 180.0
    xret = sup.pr

    d_phi = math.pi / (2.0 * n_terms)
    phi = np.array([d_phi * (i + 1) for i in range(n_terms)])
    t = np.cos(phi)

    c_of_phi = np.where(
        np.abs(t) <= xret,
        cr,
        cr + (2.0 * cr * (afil - 1.0) * (0.5 * b * t - 0.5 * b * xret) / max(b * (1.0 - xret), 1e-6)),
    )
    mu = 0.25 * c_of_phi * cla / b

    mat_b = np.zeros((n_terms, n_terms), dtype=float)
    mat_c = np.zeros(n_terms, dtype=float)
    for i in range(n_terms):
        sin_phi = math.sin(phi[i])
        for j in range(n_terms):
            m = 2 * j + 1
            mat_b[i, j] = math.sin(m * phi[i]) * (m * mu[i] + sin_phi)
        mat_c[i] = mu[i] * (aoa - aoa_zl) * sin_phi

    a_coeffs = np.linalg.solve(mat_b, mat_c)
    cl = a_coeffs[0] * math.pi * sup.ar
    cl_alpha_rad = cl / (aoa - aoa_zl)

    m_indices = np.array([2 * i + 1 for i in range(1, n_terms)], dtype=float)
    delta = float(np.sum(m_indices * (a_coeffs[1:] ** 2)) / max(a_coeffs[0] ** 2, 1e-12))
    return float(cl_alpha_rad), delta


def _build_cd_curves(
    k_factor: float,
    sup: LiftingSurface,
    aoa_stall: float,
    curva_cl: np.ndarray,
    cd_map: dict[int, float],
) -> dict[int, np.ndarray]:
    """Fit quadratic drag polars across velocities 3..15 m/s (ARRASTO.m)."""
    aoa_zl = sup.perfil.aoa_zl_rad  # Kept identical to ARRASTO.m line 11
    angulos = np.array([aoa_zl, 0.5 * (aoa_zl + aoa_stall), aoa_stall], dtype=float)
    cl_vals = np.polyval(curva_cl, angulos)
    cdi = k_factor * (cl_vals**2)
    curves: dict[int, np.ndarray] = {}
    for v_key, cd0_val in cd_map.items():
        curves[v_key] = np.polyfit(angulos, cdi + cd0_val, 2)
    return curves


def run_llt(sup: LiftingSurface) -> tuple[SurfaceAeroState, SurfaceAeroState]:
    """Compute ground-effect (SOLO) and free-flight (VOO) aerodynamic states (LLT.m)."""
    velocities = (3, 5, 7, 9, 11, 13, 15)
    cd_map = {v: cd0_completo(sup, float(v)) for v in velocities}

    cl_alpha_rad, delta = _solve_lifting_line_coeffs(sup)
    cl_a_deg = cl_alpha_rad * math.pi / 180.0
    aoa_zl_deg = sup.perfil.aoa_zl_rad * 180.0 / math.pi

    # Free-flight (LIFT_LINE.m)
    aoa_stall_voo = (sup.perfil.cl_max * 0.9) / cl_a_deg + aoa_zl_deg
    k_voo = (1.0 + delta) / (math.pi * sup.ar)
    curva_cl_voo = np.polyfit(
        [0.0, aoa_stall_voo],
        [(-aoa_zl_deg) * cl_a_deg, (aoa_stall_voo - aoa_zl_deg) * cl_a_deg],
        1,
    )
    curva_cd_voo = _build_cd_curves(k_voo, sup, aoa_stall_voo, curva_cl_voo, cd_map)
    curva_cm = np.array([0.0, sup.perfil.cm], dtype=float)

    voo_state = SurfaceAeroState(
        curva_cl=curva_cl_voo,
        curva_cm=curva_cm,
        curva_cd=curva_cd_voo,
        k_induced=k_voo,
        alfa_estol=float(aoa_stall_voo),
    )

    # Ground-effect (LIFT_LINE_SOLO.m)
    h_b = max(sup.pos[2] / max(sup.b, 1e-6), 0.01)
    afil = sup.cp / sup.cr
    ar = sup.ar
    delta_l = 1.0 - 2.25 * (afil**0.00273 - 0.997) * (ar**0.717 - 13.6)
    cl_solo_ratio = 1.0 + delta_l * (288.0 * (h_b**0.787) * math.exp(-9.14 * (h_b**0.327)) / (ar**0.882))
    cl_a_solo = cl_a_deg * cl_solo_ratio
    aoa_stall_solo = (sup.perfil.cl_max * 1.1) / cl_solo_ratio / cl_a_solo + aoa_zl_deg

    delta_d = 1.0 - 0.157 * (afil**0.775 - 0.373) * (ar**0.417 - 1.27)
    k_solo_ratio = 1.0 - delta_d * math.exp(-4.74 * (h_b**0.814)) - (h_b**2) * math.exp(-3.88 * (h_b**0.758))
    k_solo = k_voo * k_solo_ratio

    curva_cl_solo = np.polyfit(
        [0.0, aoa_stall_solo],
        [(-aoa_zl_deg) * cl_a_solo, (aoa_stall_solo - aoa_zl_deg) * cl_a_solo],
        1,
    )
    curva_cd_solo = _build_cd_curves(k_solo, sup, aoa_stall_solo, curva_cl_solo, cd_map)

    solo_state = SurfaceAeroState(
        curva_cl=curva_cl_solo,
        curva_cm=curva_cm.copy(),
        curva_cd=curva_cd_solo,
        k_induced=k_solo,
        alfa_estol=float(aoa_stall_solo),
    )
    return solo_state, voo_state


def compute_downwash(asa: LiftingSurface, sh: LiftingSurface) -> None:
    """Compute wing-induced downwash E0 and dE/dAlpha at the horizontal tail (DOWNWASH.m)."""
    assert asa.solo is not None and asa.voo is not None
    assert sh.solo is not None and sh.voo is not None

    asa.solo.cl_max = asa.solo.cl_0 + asa.solo.cl_a * asa.solo.alfa_estol
    asa.voo.cl_max = asa.voo.cl_0 + asa.voo.cl_a * asa.voo.alfa_estol
    sh.solo.cl_max = sh.solo.cl_0 + sh.solo.cl_a * sh.solo.alfa_estol
    sh.voo.cl_max = sh.voo.cl_0 + sh.voo.cl_a * sh.voo.alfa_estol

    lh = max(math.cos(math.radians(asa.inc)) * (sh.pos[0] - asa.pos[0]), 0.1)
    hh = sh.pos[2] - asa.pos[2]

    lambda_c4 = math.atan(((0.75 * asa.cr) - (0.75 * asa.cp)) / (0.5 * asa.b))
    k1_voo = (1.0 / asa.ar) - (1.0 / (1.0 + asa.ar**1.7))
    k2 = (10.0 - 3.0 * (asa.cp / asa.cr)) / 7.0
    k3 = (1.0 - (hh / asa.b)) / ((2.0 * (lh / asa.b)) ** (1.0 / 3.0))

    # Note: In MATLAB `cos(lambdac4)^1/2` evaluates as `(cos(lambdac4)^1)/2`
    base_voo = max(k1_voo * k2 * k3 * (math.cos(lambda_c4) / 2.0), 1e-6)
    e_voo = (4.44 * (base_voo**1.19)) * math.pi / 180.0
    sh.voo.e0 = math.degrees((2.0 * (asa.voo.cl_0 + asa.voo.cl_a * asa.inc)) / (math.pi * asa.ar))
    ratio_voo = max((asa.voo.cl_max / max(asa.voo.cl_0, 1e-4)) * 2.0, 0.0)
    sh.voo.de = e_voo * (ratio_voo ** (1.0 / 3.0))

    denom_solo = max(1.0 - (1.0 / (0.35**2)) * (((asa.pos[2] / asa.b) - 0.35) ** 2), 0.15)
    ar_solo = asa.ar / denom_solo
    k1_solo = (1.0 / ar_solo) - (1.0 / (1.0 + ar_solo**1.7))
    base_solo = max(k1_solo * k2 * k3 * (math.cos(lambda_c4) / 2.0), 1e-6)
    e_solo = (4.44 * (base_solo**1.19)) * math.pi / 180.0
    sh.solo.e0 = math.degrees((2.0 * (asa.solo.cl_0 + asa.solo.cl_a * asa.inc)) / (math.pi * ar_solo))
    ratio_solo = max((asa.solo.cl_max / max(asa.solo.cl_0, 1e-4)) * 2.0, 0.0)
    sh.solo.de = e_solo * (ratio_solo ** (2.0 / 3.0))


def evaluate_stability(
    asa: LiftingSurface, sh: LiftingSurface, cg: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute longitudinal stability parameters (CM_0, CM_A, PN, ME) as in AV_ESTAB.m."""
    assert asa.voo is not None and sh.voo is not None
    pos_cg_asa_x = asa.pos[0] - cg[0]
    x_c = -pos_cg_asa_x / asa.mac
    vht = (sh.s * sh.lht) / (asa.s * asa.mac)

    cm_0_asa = (asa.voo.cl_0 + asa.voo.cl_a * asa.inc) * x_c + asa.voo.cm_ca
    cm_a_asa = asa.voo.cl_a * x_c

    cm_0_sh = (
        -vht * sh.n_eff * (sh.voo.cl_0 + sh.voo.cl_a * (-sh.voo.e0 + sh.inc))
        + sh.n_eff * sh.voo.cm_ca * ((sh.mac * sh.s) / (asa.mac * asa.s))
    )
    cm_a_sh = -vht * sh.n_eff * sh.voo.cl_a * (1.0 - sh.voo.de)

    cm_0 = float(cm_0_asa + cm_0_sh)
    cm_a = float(cm_a_asa + cm_a_sh)
    me = float(-cm_a / (asa.voo.cl_a + sh.voo.cl_a * sh.n_eff * (sh.s / asa.s)))
    pn = float(me * asa.mac + cg[0])
    return cm_0, cm_a, pn, me


def _parcela_entre(x: float, min_val: float = 10.0) -> tuple[float, int]:
    """Find rib spacing between min_val and 15 cm closest to 10 cm (parcelaEntre10e15.m / parcelaEntre12e15.m)."""
    max_parcelas = max(int(math.floor(x / 10.0)), 1)
    melhor_n = 1
    melhor_parcela = x
    menor_dif = float("inf")
    for n in range(1, max_parcelas + 1):
        parcela = x / n
        if min_val <= parcela <= 15.0:
            dif = abs(parcela - 10.0)
            if dif < menor_dif:
                menor_dif = dif
                melhor_n = n
                melhor_parcela = parcela
    return melhor_parcela, melhor_n


def _pv_nervuras_surface(
    b_m: float, cr_m: float, cp_m: float, pr: float, is_wing: bool
) -> float:
    """Compute foam ribs mass [g] for wing or tail (PV_NERVURAS.m)."""
    if b_m <= 0.0 or cr_m <= 0.0:
        return 0.0
    dens_espuma = 45.0 / (1e9)  # kg/mm^3
    envergadura = b_m * 100.0
    corda_raiz = cr_m * 100.0
    corda_ponta = cp_m * 100.0

    x_asa = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    vol_asa = np.array([0.0, 10402.53, 41610.13, 93622.8, 166440.54, 260063.34])
    coefs_asa = np.polyfit(x_asa, vol_asa * dens_espuma * 1000.0, 2)

    if is_wing:
        coefs = coefs_asa
        nerv_fix = corda_raiz * (85.0 / 68.0)
        min_afil_parcela = 10.0
    else:
        x_pts = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        vol_pts = np.array([0.0, 822.26, 3289.02, 7400.3, 13156.1, 20556.4, 29601.21, 40290.54, 52624.38, 66602.73, 82225.59])
        coefs = np.polyfit(x_pts, vol_pts * dens_espuma * 1000.0, 2)
        nerv_fix = corda_raiz * (34.0 / 25.0)
        min_afil_parcela = 12.5

    semi_env_afil = (envergadura * (1.0 - pr)) / 2.0
    if semi_env_afil <= 15.0:
        melhor_parcela = max(semi_env_afil, 1.0)
    elif semi_env_afil < 25.0:
        melhor_parcela = semi_env_afil / 2.0
    else:
        melhor_parcela, _ = _parcela_entre(semi_env_afil, min_val=min_afil_parcela)

    positions = np.arange(0.0, semi_env_afil + 1e-6, melhor_parcela) if semi_env_afil > 0 else np.array([0.0])
    if semi_env_afil > 0:
        cordas_afil = corda_ponta + (semi_env_afil - positions) * (corda_raiz - corda_ponta) / semi_env_afil
        massas_afil = np.polyval(coefs, cordas_afil)
    else:
        massas_afil = np.zeros(1)

    if pr == 1.0 or abs(corda_raiz - corda_ponta) < 1e-6:
        massa_total_afil = 0.0
    elif pr == 0.0:
        massa_total_afil = float(np.sum(massas_afil[2:]) * 2.0) if len(massas_afil) > 2 else 0.0
    else:
        massa_total_afil = float(np.sum(massas_afil[1:]) * 2.0) if len(massas_afil) > 1 else 0.0

    semi_env_reta = envergadura / 2.0 if (pr == 0.0 and abs(corda_ponta - corda_raiz) < 1e-6) else (envergadura / 2.0 - semi_env_afil)
    if semi_env_reta <= 15.0:
        melhor_n = 1
    elif semi_env_reta < 25.0:
        melhor_n = 2
    else:
        _, n_calc = _parcela_entre(semi_env_reta, min_val=10.0)
        melhor_n = max(n_calc - 1, 1)

    if pr == 0.0 and abs(corda_raiz - corda_ponta) >= 1e-6:
        massa_total_reta = 0.0
    else:
        # Matches PV_NERVURAS.m line 159 which uses coefs_nerv_asa for the straight section
        massa_total_reta = float(2.0 * melhor_n * np.polyval(coefs_asa, corda_raiz))

    return float(nerv_fix + massa_total_reta + massa_total_afil)


def compute_empty_weight_pv(
    asa: LiftingSurface, sh: LiftingSurface, x_sh_ba: float, conf: int = 1, gmp: int = 1
) -> float:
    """Component buildup Empty Weight model in kg (PV_AVIAO.m)."""
    # Fuselage (PV_FUSELAGEM.m)
    coefs_fus = np.polyfit([0.0, 1160.0, 1240.0, 1600.0], [0.0, 265.0, 305.0, 350.0], 2)
    massa_fuselagem = float(np.polyval(coefs_fus, x_sh_ba * 1000.0) + 153.0 + 72.0)

    # Spar (cascao_1_long.m / cascao_1_long_biplano.m)
    coefs_long_asa = np.polyfit([0.0, 2.1, 2.5], [0.0, 75.0, 145.0], 1)
    coefs_long_sh = np.polyfit([0.0, 1.4, 2.1], [0.0, 33.0, 81.0], 1)
    massa_long_sh = float(np.polyval(coefs_long_sh, sh.b))

    # Torsion box (cascao_5_longe_de_casa_ct.m)
    massa_ct_asa = (185.0 / 1.449) * asa.s
    massa_ct_sh = (57.0 / 0.35) * sh.s

    # Ribs (PV_NERVURAS.m)
    massa_nerv_asa = _pv_nervuras_surface(asa.b, asa.cr, asa.cp, asa.pr, is_wing=True)
    massa_nerv_sh = _pv_nervuras_surface(sh.b, sh.cr, sh.cp, sh.pr, is_wing=False)

    # Covering (cascao_4_o_retorno_entelas.m)
    massa_entel_asa = 2.0 * asa.s * 40.0
    massa_entel_sh = 2.0 * sh.s * 40.0

    # Trailing edge & dowels (PV_BF_NEW.m, PV_TAR_NEW.m)
    dens_balsa = 130.0
    esp_balsa = 1.5 / 1000.0
    massa_bf_asa = 0.1 * esp_balsa * asa.b * dens_balsa * 1000.0
    massa_bf_sh = 0.021 * esp_balsa * sh.b * dens_balsa * 1000.0
    massa_tar_asa = ((6.0 / 1000.0) ** 2) * asa.b * dens_balsa * 1000.0 * 2.0
    massa_tar_sh = ((4.0 / 1000.0) ** 2) * sh.b * dens_balsa * 1000.0

    # Control surfaces (cascao_2_profundas.m, cascao_3_aileras.m)
    coefs_prof = np.polyfit([0.0, 0.35, 0.63], [0.0, 29.0, 43.0], 2)
    massa_profundas = float(np.polyval(coefs_prof, sh.s))
    coefs_ail = np.polyfit([0.0, 1.09, 1.449], [0.0, 26.0, 41.60], 1)

    if conf == 1:
        massa_long_asa = float(np.polyval(coefs_long_asa, asa.b))
        massa_asa = (
            massa_long_asa + massa_ct_asa + massa_nerv_asa + massa_entel_asa + massa_bf_asa + massa_tar_asa
        ) * 1.2102
        if asa.b > 3.0:
            massa_asa += 100.0
        massa_ailerons = float(np.polyval(coefs_ail, asa.s))
    else:
        # Biplane (conf == 2)
        massa_long_bip = asa.b * (70.0 / 2.2) * 2.0
        massa_asa = (
            massa_long_bip
            + 2.0 * massa_ct_asa
            + 2.0 * massa_entel_asa
            + 2.0 * massa_nerv_asa
            + 2.0 * massa_bf_asa
            + 2.0 * massa_tar_asa
        ) * 1.4411
        if asa.b > 3.0:
            massa_asa += 200.0
        massa_ailerons = float(np.polyval(coefs_ail, 2.0 * asa.s))

    massa_sh = (
        massa_long_sh + massa_ct_sh + massa_nerv_sh + massa_entel_sh + massa_bf_sh + massa_tar_sh
    ) * 1.11

    if gmp == 1:
        val_motor, val_helice = 255.0, 231.0
    elif gmp == 2:
        val_motor, val_helice = 255.0, 96.0
    else:
        val_motor, val_helice = 450.0, 214.0

    fixed_avionics_and_hardware = (
        40.0  # EV
        + 250.0  # Bateria
        + 100.0  # ESC
        + 39.0  # Armplug
        + 83.0  # Wattimetro
        + 46.0  # Bat pel
        + 15.0  # Receptor
        + 123.20  # Servos
        + 15.0  # Links
        + 8.0  # Chave
        + (10.0 * x_sh_ba)  # Cablagem
        + 16.0  # Tampa
        + 9.0  # Voltwatch
        + 2.50  # Regulador
        + 46.30  # Parafusos
        + 58.0  # Limitador
    )
    pv_grams = (
        massa_asa
        + massa_sh
        + massa_profundas
        + massa_ailerons
        + massa_fuselagem
        + val_motor
        + val_helice
        + fixed_avionics_and_hardware
    )
    return float(pv_grams / 1000.0)


def coef_curva_xf(state: SurfaceAeroState, aoa_deg: float, v: float) -> tuple[float, float, float]:
    """Interpolate CL, CD, CM across velocity brackets (COEF_CURVA_XF.m)."""
    cl = float(np.polyval(state.curva_cl, aoa_deg))
    cm = float(state.cm_ca)

    brackets = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15)]
    if v <= 3.0:
        cd = float(np.polyval(state.curva_cd[3], aoa_deg))
    elif v > 15.0:
        cd = float(np.polyval(state.curva_cd[15], aoa_deg))
    else:
        cd = float(np.polyval(state.curva_cd[15], aoa_deg))
        for v_low, v_high in brackets:
            if v_low < v <= v_high:
                cd_min = float(np.polyval(state.curva_cd[v_low], aoa_deg))
                cd_max = float(np.polyval(state.curva_cd[v_high], aoa_deg))
                cd = cd_min + ((v - v_low) / 2.0) * (cd_max - cd_min)
                break

    cd = abs(cd) * 1.30  # Calibration factor in COEF_CURVA_XF.m line 80
    return cl, cd, cm


def tracao(v: float, gmp: int = 1) -> float:
    """Propulsion thrust curve T(v) [N] (TRACAO.m)."""
    if gmp == 2:
        return 0.0092 * (v**2) - 1.3516 * v + 40.2471
    return 0.0098 * (v**2) - 1.433 * v + 42.739


def _trimado_desemp(
    asa: LiftingSurface,
    sh: LiftingSurface,
    cg: np.ndarray,
    tra_pos: np.ndarray,
    peso: float,
    v: float,
    rho: float,
    gmp: int,
) -> tuple[float, float, float]:
    """Iterative longitudinal trim solver in climb/post-obstacle (TRIMADO_DESEMP.m)."""
    assert asa.voo is not None and sh.voo is not None
    incl = 0.6
    precisao = 0.01

    pos_asa = np.array([asa.pos[0] - cg[0], asa.pos[2] - cg[2]])
    pos_sh = np.array([sh.pos[0] - cg[0], sh.pos[2] - cg[2]])
    pos_tra = np.array([tra_pos[0] - cg[0], tra_pos[2] - cg[2]])

    dis_tra = math.hypot(pos_tra[0], pos_tra[1])
    dis_asa = math.hypot(pos_asa[0], pos_asa[1])
    dis_emp = math.hypot(pos_sh[0], pos_sh[1])

    theta_tra_1 = math.degrees(math.atan2(pos_tra[1], max(-pos_tra[0], 1e-6)))
    theta_asa_1 = math.degrees(math.atan2(pos_asa[1], max(-pos_asa[0], 1e-6)))
    theta_emp_1 = math.degrees(math.atan2(pos_sh[1], max(pos_sh[0], 1e-6)))

    q = 0.5 * rho * (v**2)
    tra = tracao(v, gmp)
    aoa = 0.0
    prof = 0.0
    mt_prev = 1.0

    for _ in range(40):
        th_tra = theta_tra_1 + aoa - asa.inc
        th_asa = theta_asa_1 + aoa - asa.inc
        th_emp = theta_emp_1 - aoa + asa.inc

        x_tra = dis_tra * math.cos(math.radians(th_tra))
        h_tra = dis_tra * math.sin(math.radians(th_tra))
        x_asa = dis_asa * math.cos(math.radians(th_asa))
        h_asa = dis_asa * math.sin(math.radians(th_asa))
        x_emp = dis_emp * math.cos(math.radians(th_emp))
        h_emp = dis_emp * math.sin(math.radians(th_emp))

        _, cd_asa, _ = coef_curva_xf(asa.voo, aoa, v)
        d_asa = cd_asa * q * asa.s
        mca_asa = asa.voo.cm_ca * q * asa.s * asa.mac

        _, cd_sh, _ = coef_curva_xf(sh.voo, prof, v)
        d_sh = sh.n_eff * cd_sh * q * sh.s
        mca_sh = sh.n_eff * sh.voo.cm_ca * q * sh.s * sh.mac

        peso_z = peso * math.cos(math.radians(incl))
        tra_z = tra * math.sin(math.radians(aoa))
        tra_x = tra * math.cos(math.radians(aoa))

        m_d = d_asa * h_asa + d_sh * h_emp
        m_p_sh = peso_z * x_emp
        m_tra = tra_x * h_tra - tra_z * x_tra - tra_z * x_emp
        m_sup = mca_asa + mca_sh

        denom_x = x_asa + x_emp
        if abs(denom_x) < 1e-6:
            break
        l_wing = (m_tra - m_d - m_sup + m_p_sh) / denom_x
        l_emp = peso_z - l_wing - tra_z
        m_l = l_wing * x_asa - l_emp * x_emp

        cl_wing = (2.0 * l_wing) / (rho * (v**2) * asa.s)
        cl_emp = (2.0 * l_emp) / (rho * (v**2) * sh.s)

        aoa = (cl_wing - asa.voo.cl_0) / max(asa.voo.cl_a, 1e-6)
        prof = (cl_emp - sh.voo.cl_0) / max(sh.voo.cl_a, 1e-6)

        _, cd_asa2, _ = coef_curva_xf(asa.voo, aoa, v)
        _, cd_sh2, _ = coef_curva_xf(sh.voo, prof, v)
        d_total = cd_asa2 * q * asa.s + sh.n_eff * cd_sh2 * q * sh.s
        mt_curr = (cd_asa2 * q * asa.s * h_asa + sh.n_eff * cd_sh2 * q * sh.s * h_emp) + m_sup + m_l - tra_x * h_tra + tra_z * x_tra

        if abs(mt_curr) < precisao or abs(abs(mt_curr) - abs(mt_prev)) < 1e-3 or abs(mt_curr) > 1e9:
            break
        mt_prev = mt_curr

    _, cd_asa_f, _ = coef_curva_xf(asa.voo, aoa, v)
    _, cd_sh_f, _ = coef_curva_xf(sh.voo, prof, v)
    d_final = cd_asa_f * q * asa.s + sh.n_eff * cd_sh_f * q * sh.s
    cos_aoa = max(math.cos(math.radians(np.clip(aoa, -45.0, 45.0))), 0.2)
    tra_exc = tra - (d_final / cos_aoa)
    roc = (tra_exc * v) / peso
    return float(aoa), float(prof), float(roc)


def simulate_takeoff_corrida(
    asa: LiftingSurface,
    sh: LiftingSurface,
    cg: np.ndarray,
    tdp_pos: np.ndarray,
    tdn_pos: np.ndarray,
    tra_pos: np.ndarray,
    gmp: int = 1,
) -> dict[str, float]:
    """Bisection MTOW search with time-stepped ground roll & rotation (MDO__CORRIDA_AOC_OFICIAL.m)."""
    assert asa.solo is not None and asa.voo is not None
    assert sh.solo is not None and sh.voo is not None

    iy = 0.5
    rho = 1.1
    g = 9.81
    dt = 0.01
    c_at = 0.07
    incl = 0.6
    fator_vprof = 1.05
    prof_mao = 5.0

    x_max_dec = 50.0
    x_obs = 55.0
    h_tr = 0.9 * 1.1

    precis = 0.02
    teto = 20.0
    piso = 6.0  # Extended lower bound so light geometries still return accurate MTOW

    pos_pasa = np.array([asa.pos[0] - tdp_pos[0], asa.pos[2] - tdp_pos[2]])
    pos_psh = np.array([sh.pos[0] - tdp_pos[0], sh.pos[2] - tdp_pos[2]])
    pos_pcg = np.array([cg[0] - tdp_pos[0], cg[2] - tdp_pos[2]])
    pos_ptra = np.array([tra_pos[0] - tdp_pos[0], tra_pos[2] - tdp_pos[2]])
    pos_tdn_x = tdn_pos[0] - tdp_pos[0]

    best_x_dec = 55.0
    best_v_dec = 10.0
    best_v_prof = 10.0
    chute = piso

    sh_ang_max = (
        (asa.solo.alfa_estol - (asa.inc - sh.inc))
        - sh.solo.de * asa.solo.alfa_estol
        - sh.solo.e0
        - prof_mao
    )
    cl_max_asa = float(np.polyval(asa.solo.curva_cl, asa.solo.alfa_estol))
    cl_max_sh = float(np.polyval(sh.solo.curva_cl, sh_ang_max))
    cl_max_global = max(cl_max_asa + cl_max_sh * (sh.s / asa.s), 0.2)

    if sh_ang_max >= sh.solo.alfa_estol:
        return {"MTOW": piso, "PROF_DEC": prof_mao, "X_DEC": 55.0, "V_DEC": 10.0, "V_PROF": 10.0}

    while abs(piso - teto) >= precis:
        chute = 0.5 * (piso + teto)
        m = chute
        peso = m * g
        peso_z = peso * math.cos(math.radians(incl))
        peso_x = peso * math.sin(math.radians(incl))

        v_stall = math.sqrt((2.0 * peso) / (rho * cl_max_global * asa.s))
        v_profundor = fator_vprof * v_stall

        v = 0.0
        s = 0.0
        aoa = 0.0
        failed = False

        # Phase 1: Ground roll before elevator deflection (matches MDO__CORRIDA_AOC_OFICIAL.m lines 98-218)
        # Initial impulse at i=1 (v=0 -> v(1)=a(1)*dt, s(1)=0)
        tra0 = tracao(0.0, gmp)
        a_prev = g * (tra0 - c_at * peso_z - peso_x) / peso
        if a_prev <= 0.0:
            teto = chute
            continue
        v_curr = a_prev * dt
        s_curr = 0.0
        v_prev = 0.0
        s_prev = 0.0
        l_asa = 0.0
        l_sh = 0.0
        q = 0.0

        while v_curr <= v_profundor:
            v_prev = v_curr
            s_prev = s_curr
            q = 0.5 * (v_prev**2) * rho
            aoa_asa = asa.inc
            cl_asa, cd_asa, _ = coef_curva_xf(asa.solo, aoa_asa, v_prev)
            l_asa = cl_asa * q * asa.s
            d_asa = cd_asa * q * asa.s

            aoa_sh = sh.inc - sh.solo.e0 - sh.solo.de * aoa_asa
            cl_sh, cd_sh, _ = coef_curva_xf(sh.solo, aoa_sh, v_prev)
            l_sh = sh.n_eff * cl_sh * q * sh.s
            d_sh = sh.n_eff * cd_sh * q * sh.s

            tra = tracao(v_prev, gmp)
            a_curr = g * (tra - d_asa - d_sh - c_at * (peso_z - l_asa - l_sh) - peso_x) / peso
            v_curr = v_prev + a_curr * dt
            s_curr = s_prev + v_prev * dt + 0.5 * a_curr * (dt**2)

            if a_prev <= 0.0 or s_prev > x_max_dec:
                teto = chute
                failed = True
                break
            a_prev = a_curr

        if failed:
            continue

        # Phase 2: Initial elevator deflection at index i (lines 237-280)
        iy_tdp = iy + m * (pos_pcg[1] ** 2)
        aoa_asa = asa.inc
        cl_asa, cd_asa, _ = coef_curva_xf(asa.solo, aoa_asa, v_prev)
        l_asa = cl_asa * q * asa.s
        d_asa = cd_asa * q * asa.s
        mca_asa = asa.solo.cm_ca * q * asa.s * asa.mac

        aoa_sh = sh.inc - sh.solo.e0 - sh.solo.de * aoa_asa - prof_mao
        cl_sh, cd_sh, _ = coef_curva_xf(sh.solo, aoa_sh, v_prev)
        l_sh = sh.n_eff * cl_sh * q * sh.s
        d_sh = sh.n_eff * cd_sh * q * sh.s
        mca_sh = sh.n_eff * sh.solo.cm_ca * q * sh.s * sh.mac

        tra = tracao(v_prev, gmp)
        m_asa = (-l_asa * pos_pasa[0]) + (d_asa * pos_pasa[1]) + mca_asa
        m_sh = (-l_sh * pos_psh[0]) + (d_sh * pos_psh[1]) + mca_sh
        m_tra = -tra * pos_ptra[1]
        m_peso = (peso_z * pos_pcg[0]) + (peso_x * pos_pcg[1])
        m_res = m_asa + m_sh + m_tra + m_peso

        a_ang = m_res / iy_tdp
        v_ang = a_ang * dt
        ang_rot = 0.5 * a_ang * (dt**2)
        aoa_prev = math.degrees(ang_rot)
        aoa = aoa_prev
        fz_res = (l_asa + l_sh) - peso

        while fz_res <= 0.0:
            v_prev = v_curr
            s_prev = s_curr
            aoa_prev = aoa
            q = 0.5 * (v_prev**2) * rho
            cos_a = math.cos(math.radians(aoa_prev))
            sin_a = math.sin(math.radians(aoa_prev))
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            tra = tracao(v_prev, gmp)
            aoa_asa = aoa_prev + asa.inc
            if aoa_asa >= asa.solo.alfa_estol:
                teto = chute
                failed = True
                break

            cl_asa, cd_asa, _ = coef_curva_xf(asa.solo, aoa_asa, v_prev)
            l_asa = cl_asa * q * asa.s
            d_asa = cd_asa * q * asa.s
            mca_asa = asa.solo.cm_ca * q * asa.s * asa.mac

            aoa_sh = aoa_prev + sh.inc - sh.solo.e0 - sh.solo.de * aoa_asa - prof_mao
            cl_sh, cd_sh, _ = coef_curva_xf(sh.solo, aoa_sh, v_prev)
            l_sh = sh.n_eff * cl_sh * q * sh.s
            d_sh = sh.n_eff * cd_sh * q * sh.s
            mca_sh = sh.n_eff * sh.solo.cm_ca * q * sh.s * sh.mac

            p_asa_rot = pos_pasa @ rot
            p_sh_rot = pos_psh @ rot
            p_cg_rot = pos_pcg @ rot

            m_asa = (-l_asa * p_asa_rot[0]) + (d_asa * p_asa_rot[1]) + mca_asa
            m_sh = (-l_sh * p_sh_rot[0]) + (d_sh * p_sh_rot[1]) + mca_sh
            m_tra = -tra * pos_ptra[1]
            m_peso = (peso_z * p_cg_rot[0]) + (peso_x * p_cg_rot[1])
            m_res = m_asa + m_sh + m_tra + m_peso

            a_ang = m_res / iy_tdp
            ang_rot += v_ang * dt + 0.5 * a_ang * (dt**2)
            v_ang += a_ang * dt
            aoa = math.degrees(ang_rot)

            a_curr = g * (tra - d_asa - d_sh - c_at * (peso_z - l_asa - l_sh) - peso_x) / peso
            s_curr = s_prev + v_prev * dt + 0.5 * a_curr * (dt**2)
            v_curr = v_prev + a_curr * dt
            fz_res = l_asa + l_sh - peso

            if a_prev <= 0.0 or s_prev > x_max_dec or aoa < -10.0:
                teto = chute
                failed = True
                break
            a_prev = a_curr
            if fz_res > 0.0:
                break

        if failed:
            continue

        s = s_curr
        v = v_curr

        # Phase 3: Obstacle clearance & climb check (lines 389-415)
        aoa_asa = aoa_prev + asa.inc
        _, cd_asa_voo, _ = coef_curva_xf(asa.voo, aoa_asa, v)
        aoa_sh = aoa_prev + sh.inc - sh.voo.e0 - sh.voo.de * aoa_asa - prof_mao
        _, cd_sh_voo, _ = coef_curva_xf(sh.voo, aoa_sh, v)
        d_voo = cd_asa_voo * q * asa.s + sh.n_eff * cd_sh_voo * q * sh.s
        tra = tracao(v_prev, gmp)
        roc_sub = ((tra - d_voo) * v) / peso
        aoc_sub = math.degrees(math.atan2(roc_sub, max(v, 1e-3)))
        x_tr = max(x_obs - s, 0.5)
        aoc_rec_sub = math.degrees(math.atan2(h_tr, x_tr))

        # Phase 4: Post-obstacle trim check
        aoa_asa_trim, aoa_sh_trim, _ = _trimado_desemp(
            asa, sh, cg, tra_pos, peso, v_profundor, rho, gmp
        )
        if aoa_asa_trim >= asa.voo.alfa_estol or math.isnan(aoa_asa_trim):
            teto = chute
            continue

        q_pos = 0.5 * rho * (v_profundor**2)
        _, cd_asa_pos, _ = coef_curva_xf(asa.voo, aoa_asa_trim, v_profundor)
        _, cd_sh_pos, _ = coef_curva_xf(sh.voo, aoa_sh_trim, v_profundor)
        d_pos = cd_asa_pos * q_pos * asa.s + sh.n_eff * cd_sh_pos * q_pos * sh.s
        cos_trim = max(math.cos(math.radians(np.clip(aoa_asa_trim, -40.0, 40.0))), 0.2)
        tra_exc_pos = tracao(v_profundor, gmp) - (d_pos / cos_trim)
        roc_pos = (tra_exc_pos * v_profundor) / peso

        if aoc_sub < aoc_rec_sub or roc_pos < 0.0:
            teto = chute
            continue

        piso = chute
        best_x_dec = s
        best_v_dec = v
        best_v_prof = v_profundor

    return {
        "MTOW": float(chute),
        "PROF_DEC": float(prof_mao),
        "X_DEC": float(best_x_dec),
        "V_DEC": float(best_v_dec),
        "V_PROF": float(best_v_prof),
    }


def compute_flight_score_pvoo(cp: float, pv: float, b_asa: float, n_surfaces: int = 1) -> float:
    """Compute PVOO according to Eq. 1.1 of Relatório 2026 and line 348 of MDO__TRESIF_MONOPLANO.m.

    PVOO = 50 * EE * f(N) * 1.15^(-b_w), where f(N) = 0.185*N^2 - 0.775*N + 1.81.
    Note: MDO__TRESIF_MONOPLANO.m line 348 uses N=2 in f(N) even for CONF=1; here we use N=1 for monoplane
    and N=2 for biplane as defined in Eq. 1.1 of Relatório 2026.
    """
    if pv <= 0.0 or cp <= 0.0:
        return 0.0
    ee = cp / pv
    f_n = 0.185 * (n_surfaces**2) - 0.775 * n_surfaces + 1.81
    return float((ee * 50.0) * f_n * (1.15 ** (-b_asa)))


def run_simulation(inp: AircraftInput) -> SimulationResult:
    """Run complete coupled aerodynamic, stability, empty weight, and takeoff simulation."""
    t0 = time.perf_counter()
    is_biplane = inp.architecture == "biplane"
    conf = 2 if is_biplane else 1
    n_surfaces = 2 if is_biplane else 1

    r_tdp = 0.030 if conf == 1 else 0.055
    x_tdn = -0.01 if conf == 1 else -0.10
    x_tra = -0.35 if conf == 1 else -0.20
    helc = 0.6858 if inp.gmp == 1 else 0.508

    z_asa2 = inp.z_asa2 if (is_biplane and inp.z_asa2 is not None) else (inp.z_asa + 0.25)
    gap = max(z_asa2 - inp.z_asa, 0.10) if is_biplane else 0.0

    asa = LiftingSurface(
        name="ASA",
        b=inp.b_asa,
        cr=inp.cr_asa,
        cp=inp.cr_asa * inp.afil_asa,
        pr=inp.pr_asa,
        inc=inp.inc_asa,
        pos_ba=np.array([0.0, 0.0, inp.z_asa]),
        pos=np.zeros(3),
        perfil=get_airfoil_profile(1),
        gap=gap,
    )
    compute_geometry(asa)

    cg = np.array([inp.x_cg * asa.mac, 0.0, inp.z_asa])
    tdp_pos = np.array([cg[0] + 0.01, 0.0, r_tdp])
    tdn_pos = np.array([x_tdn, 0.0, r_tdp])
    tra_pos = np.array([x_tra, 0.0, 0.5 * helc + 0.02])

    asa.pos = np.array([0.25 * asa.mac, 0.0, inp.z_asa])

    sh = LiftingSurface(
        name="SH",
        b=inp.b_sh,
        cr=inp.cr_sh,
        cp=inp.cr_sh * inp.afil_sh,
        pr=inp.pr_sh,
        inc=inp.inc_sh,
        pos_ba=np.array([inp.x_sh_ba, 0.0, inp.z_sh]),
        pos=np.zeros(3),
        perfil=get_airfoil_profile(2),
    )
    compute_geometry(sh)
    sh.pos = np.array([inp.x_sh_ba + 0.25 * sh.mac, 0.0, inp.z_sh])
    sh.lht = float(sh.pos[0] - cg[0])

    asa_aero_target = apply_equivalent_monoplane(asa, gap) if is_biplane else asa
    asa.solo, asa.voo = run_llt(asa_aero_target)
    sh.solo, sh.voo = run_llt(sh)

    # Evaluate CM_CA, CL_0, CL_A
    for sup in (asa, sh):
        assert sup.solo is not None and sup.voo is not None
        sup.solo.cm_ca = float(np.polyval(sup.solo.curva_cm, sup.inc))
        sup.voo.cm_ca = float(np.polyval(sup.voo.curva_cm, sup.inc))
        sup.solo.cl_0 = float(np.polyval(sup.solo.curva_cl, 0.0))
        sup.solo.cl_a = float(
            (np.polyval(sup.solo.curva_cl, sup.solo.alfa_estol) - sup.solo.cl_0) / sup.solo.alfa_estol
        )
        sup.voo.cl_0 = float(np.polyval(sup.voo.curva_cl, 0.0))
        sup.voo.cl_a = float(
            (np.polyval(sup.voo.curva_cl, sup.voo.alfa_estol) - sup.voo.cl_0) / sup.voo.alfa_estol
        )

    compute_downwash(asa, sh)
    assert asa.solo is not None and asa.voo is not None
    assert sh.solo is not None and sh.voo is not None

    alfa_solo_max = asa.solo.alfa_estol - asa.inc
    sh.solo.alfa_max = alfa_solo_max + sh.inc - sh.solo.e0 - sh.solo.de * alfa_solo_max
    asa.solo.cl_max = asa.solo.cl_0 + asa.solo.cl_a * asa.solo.alfa_estol
    sh.solo.cl_max = sh.solo.cl_0 + sh.solo.cl_a * sh.solo.alfa_max
    cl_max_solo = float(asa.solo.cl_max + sh.solo.cl_max * (sh.s / asa.s))

    alfa_voo_max = asa.voo.alfa_estol - asa.inc
    sh.voo.alfa_max = alfa_voo_max + sh.inc - sh.voo.e0 - sh.voo.de * alfa_voo_max
    asa.voo.cl_max = asa.voo.cl_0 + asa.voo.cl_a * asa.voo.alfa_estol
    sh.voo.cl_max = sh.voo.cl_0 + sh.voo.cl_a * sh.voo.alfa_max
    cl_max_voo = float(asa.voo.cl_max + sh.voo.cl_max * (sh.s / asa.s))

    cm_0, cm_a, _, me = evaluate_stability(asa, sh, cg)
    corrida = simulate_takeoff_corrida(asa, sh, cg, tdp_pos, tdn_pos, tra_pos, gmp=inp.gmp)

    mtow = corrida["MTOW"]
    pv = compute_empty_weight_pv(asa, sh, inp.x_sh_ba, conf=conf, gmp=inp.gmp)
    cp = max(mtow - pv, 0.0)
    ee = cp / max(pv, 1e-6)
    p_voo = compute_flight_score_pvoo(cp, pv, asa.b, n_surfaces=n_surfaces)

    top_wing_z = (inp.z_asa + gap) if is_biplane else inp.z_asa
    h_total = float(max(tra_pos[2] + 0.5 * helc, top_wing_z, sh.pos[2]))

    # Constraints (MDO__FUNCAO_OBJETIVO.m lines 83-96 & Table 4.4)
    g_vec = [
        4.1 - asa.ar,
        0.0 - cm_0,
        0.0 + cm_a,
        0.02 - me,
        -0.9 + h_total,
        -4.0 + asa.b,
        4.1 - sh.ar,
    ]
    rp = 1e-6
    penalty = float((1.0 / rp) * sum(max(0.0, g_i) ** 2 for g_i in g_vec))

    violated: list[str] = []
    if asa.ar < 4.1:
        violated.append("REQ-AEROD-001 (AR_ASA < 4.1)")
    if sh.ar < 4.1:
        violated.append("REQ-AEROD-001 (AR_SH < 4.1)")
    if cm_0 < 0.0:
        violated.append("REQ-ESTC-015 (CM_0 < 0)")
    if cm_a >= 0.0:
        violated.append("REQ-ESTC-012 (CM_A >= 0)")
    if me < 0.02:
        violated.append("REQ-ESTC-012 (ME < 0.02)")
    if h_total > 0.9:
        violated.append("REQ-SYS-005 (H_TOTAL > 0.9 m)")
    if asa.b > 4.0:
        violated.append("REQ-SYS-001 (b_ASA > 4.0 m)")
    if corrida["X_DEC"] > 55.0:
        violated.append("REQ-ADES-002 (X_DEC > 55 m)")

    _, cd_cruise, _ = coef_curva_xf(asa.voo, 2.0, 13.0)
    total_s_wing = (2.0 * asa.s) if is_biplane else asa.s
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SimulationResult(
        inputs=inp,
        constraints=SimulationConstraints(
            ar_asa=round(asa.ar, 4),
            ar_sh=round(sh.ar, 4),
            cm_0=round(cm_0, 5),
            cm_a=round(cm_a, 5),
            me=round(me, 5),
            h_total=round(h_total, 4),
            b_asa=round(asa.b, 4),
            x_dec=round(corrida["X_DEC"], 3),
            penalty=round(penalty, 3),
            is_feasible=(len(violated) == 0 and mtow > pv + 1.0),
            violated_constraints=violated,
        ),
        outputs=SimulationOutputs(
            mtow=round(mtow, 4),
            pv=round(pv, 4),
            cp=round(cp, 4),
            ee=round(ee, 4),
            p_voo=round(p_voo, 4),
            objective_f=round(-p_voo + penalty, 4),
            v_dec=round(corrida["V_DEC"], 4),
            x_dec=round(corrida["X_DEC"], 4),
            cl_max_solo=round(cl_max_solo, 4),
            cl_max_voo=round(cl_max_voo, 4),
            cd_voo_cruise=round(cd_cruise, 5),
            estol_av_solo=round(alfa_solo_max, 3),
            estol_av_voo=round(alfa_voo_max, 3),
            s_asa=round(total_s_wing, 4),
            s_sh=round(sh.s, 4),
            lht=round(sh.lht, 4),
            wing_loading=round(mtow / max(total_s_wing, 1e-6), 4),
        ),
        solver="python_llt",
        execution_time_ms=round(elapsed_ms, 3),
    )


def sample_doe_lhs(
    n_samples: int = 100,
    architecture: str = "monoplane",
    seed: int = 42,
) -> list[AircraftInput]:
    """Sample aircraft design space using Latin Hypercube Sampling (Sec. 4.2 & Table 4.5, Relatório 2026)."""
    sampler = qmc.LatinHypercube(d=14, seed=seed)
    u = sampler.random(n=n_samples)

    # Bounds from MDO__FUNCAO_OBJETIVO.m & Table 4.5
    bounds = np.array(
        [
            [0.15, 0.55],  # x_cg
            [1.60, 3.80],  # b_asa
            [0.15, 0.30],  # z_asa
            [0.50, 0.70],  # afil_asa
            [0.35, 0.75],  # cr_asa
            [0.30, 0.70],  # pr_asa
            [-2.0, 2.0],   # inc_asa (rounded)
            [0.60, 1.60],  # b_sh
            [1.20, 2.00],  # x_sh_ba
            [0.20, 0.60],  # z_sh
            [0.50, 0.90],  # afil_sh
            [0.15, 0.35],  # cr_sh
            [0.30, 0.80],  # pr_sh
            [-3.0, 1.0],   # inc_sh (rounded)
        ]
    )
    scaled = qmc.scale(u, bounds[:, 0], bounds[:, 1])
    samples: list[AircraftInput] = []
    for row in scaled:
        z_asa = float(row[2])
        samples.append(
            AircraftInput(
                architecture="biplane" if architecture == "biplane" else "monoplane",
                x_cg=round(float(row[0]), 4),
                b_asa=round(float(row[1]), 4),
                z_asa=round(z_asa, 4),
                z_asa2=round(z_asa + 0.28, 4) if architecture == "biplane" else None,
                afil_asa=round(float(row[3]), 4),
                cr_asa=round(float(row[4]), 4),
                pr_asa=round(float(row[5]), 4),
                inc_asa=float(round(row[6])),
                b_sh=round(float(row[7]), 4),
                x_sh_ba=round(float(row[8]), 4),
                z_sh=round(float(row[9]), 4),
                afil_sh=round(float(row[10]), 4),
                cr_sh=round(float(row[11]), 4),
                pr_sh=round(float(row[12]), 4),
                inc_sh=float(round(row[13])),
                gmp=1,
            )
        )
    return samples
