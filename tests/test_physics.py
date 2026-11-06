"""Unit tests for the coupled AeroDesign physics simulator and DoE sampler."""

from __future__ import annotations

from aeromdo.physics import compute_flight_score_pvoo, run_simulation, sample_doe_lhs
from aeromdo.schemas import AircraftInput


def test_pvoo_equation_1_1() -> None:
    """Verify Eq. 1.1 from Relatório 2026: PVOO = 50 * (CP/PV) * f(N) * 1.15^(-bw)."""
    cp = 10.5
    pv = 3.0
    b_asa = 2.26
    score_mono = compute_flight_score_pvoo(cp=cp, pv=pv, b_asa=b_asa, n_surfaces=1)
    score_bip = compute_flight_score_pvoo(cp=cp, pv=pv, b_asa=b_asa, n_surfaces=2)
    assert score_mono > 120.0
    assert score_bip < score_mono  # f(1)=1.22 > f(2)=1.00


def test_monoplane_and_biplane_simulation() -> None:
    """Ensure both monoplane and biplane simulations run and produce physically consistent outputs."""
    mono_in = AircraftInput(
        architecture="monoplane",
        x_cg=0.32,
        b_asa=2.35,
        z_asa=0.20,
        afil_asa=0.58,
        cr_asa=0.60,
        pr_asa=0.50,
        inc_asa=0.0,
        b_sh=1.10,
        x_sh_ba=1.55,
        z_sh=0.38,
        afil_sh=0.65,
        cr_sh=0.24,
        pr_sh=0.50,
        inc_sh=-1.0,
    )
    res_mono = run_simulation(mono_in)
    assert 6.0 <= res_mono.outputs.mtow <= 20.0
    assert 1.5 <= res_mono.outputs.pv <= 6.0
    assert res_mono.constraints.ar_asa >= 3.5
    assert res_mono.constraints.cm_a < 0.0

    bip_in = mono_in.model_copy(update={"architecture": "biplane", "z_asa2": 0.48})
    res_bip = run_simulation(bip_in)
    assert res_bip.outputs.pv > res_mono.outputs.pv
    assert res_bip.outputs.s_asa > res_mono.outputs.s_asa


def test_doe_lhs_sampling() -> None:
    """Verify Latin Hypercube Sampling generates diverse candidates within Table 4.5 bounds."""
    samples = sample_doe_lhs(n_samples=25, architecture="monoplane", seed=7)
    assert len(samples) == 25
    spans = [s.b_asa for s in samples]
    assert min(spans) >= 1.60
    assert max(spans) <= 3.80
    assert len(set(spans)) == 25


def test_octave_matlab_numerical_parity() -> None:
    """Verify Python physics port matches GNU Octave 11.3.0 running MDO__TRESIF_MONOPLANO.m."""
    inp = AircraftInput(
        x_cg=0.30,
        b_asa=2.35,
        z_asa=0.22,
        afil_asa=0.55,
        cr_asa=0.62,
        pr_asa=0.45,
        inc_asa=1.5,
        b_sh=1.15,
        x_sh_ba=1.65,
        z_sh=0.42,
        afil_sh=0.70,
        cr_sh=0.28,
        pr_sh=0.50,
        inc_sh=-1.5,
        gmp=1,
    )
    res = run_simulation(inp)
    o = res.outputs
    c = res.constraints

    # Octave 11.3.0 ground truth values from MDO__TRESIF_MONOPLANO.m
    assert abs(o.mtow - 11.650391) < 0.01
    assert abs(o.pv - 2.890589) < 0.001
    assert abs(o.cl_max_voo - 1.589533) < 0.0005
    assert abs(o.cl_max_solo - 1.643721) < 0.0005
    assert abs(c.cm_0 - 0.072176) < 0.0005
    assert abs(c.cm_a - (-0.038022)) < 0.0005
    assert abs(c.me - 0.453498) < 0.0005
    assert abs(o.s_asa - 1.276696) < 0.0005
    assert abs(c.ar_asa - 4.325618) < 0.0005
    assert abs(o.lht - 1.547847) < 0.0005

