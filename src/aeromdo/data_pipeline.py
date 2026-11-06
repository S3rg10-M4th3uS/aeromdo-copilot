"""Data ingestion and normalization pipeline for MDO spreadsheets (~51k evaluations) and 2026 PDF Report."""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromdo.physics import compute_empty_weight_pv, LiftingSurface, compute_geometry, get_airfoil_profile


DEFAULT_RAW_DIR = Path("/home/matheus/workspace/pessoal/MONOPLANO/EOCORRIDAS2024_CalibradoXF_avofc_rela")
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "aeromdo.db"


COLUMN_ALIASES: dict[str, str] = {
    "MTOW": "mtow",
    "F": "objective_f",
    "P": "penalty",
    "ATTE": "penalty",
    "INPUT.POTELE": "pot_ele",
    " INPUT.POTELE": "pot_ele",
    "INPUT.b_ASA": "b_asa",
    "INPUT.Z_ASA": "z_asa",
    "INPUT.Z_ASA2": "z_asa2",
    "INPUT.AFIL_ASA": "afil_asa",
    "INPUT.CR_ASA": "cr_asa",
    "INPUT.PR_ASA": "pr_asa",
    "INPUT.INC_ASA": "inc_asa",
    "INPUT.b_SH": "b_sh",
    "INPUT.X_SH_BA": "x_sh_ba",
    "INPUT.Z_SH": "z_sh",
    "INPUT.AFIL_SH": "afil_sh",
    "INPUT.CR_SH": "cr_sh",
    "INPUT.PR_SH": "pr_sh",
    "INPUT.INC_SH": "inc_sh",
    "CONSTRANTS.AR_ASA": "ar_asa",
    "CONSTRANTS.AR_SH": "ar_sh",
    "CONSTRANTS.CM_0": "cm_0",
    "CONSTRANTS.CM_A": "cm_a",
    "CONSTRANTS.ME": "me",
    "CONSTRANTS.GEOMETRIA.H": "h_total",
    "CONSTRANTS.GEOMETRIA.SOMA": "geom_soma",
    "CONSTRANTS.GEOMETRIA.ResFis": "geom_res_fis",
    "OUTPUT.V_DEC": "v_dec",
    "OUTPUT.X_DEC": "x_dec",
    "OUTPUT.CL_MAX_SOLO": "cl_max_solo",
    "OUTPUT.CL_MAX_VOO": "cl_max_voo",
    "OUTPUT.ESTOL_AV_SOLO": "estol_av_solo",
    "OUTPUT.ESTOL_AV_VOO": "estol_av_voo",
    "OUTPUT.ESTOL_ASA_SOLO": "estol_asa_solo",
    "OUTPUT.ESTOL_ASA_VOO": "estol_asa_voo",
    "OUTPUT.ESTOL_SH_SOLO": "estol_sh_solo",
    "OUTPUT.ESTOL_SH_VOO": "estol_sh_voo",
    "OUTPUT.X_CG": "x_cg_m",
    "OUTPUT.Z_CG": "z_cg_m",
    "OUTPUT.S_ASA": "s_asa",
    "OUTPUT.S_SH": "s_sh",
    "OUTPUT.AR_ASA": "out_ar_asa",
    "OUTPUT.AR_SH": "out_ar_sh",
    "OUTPUT.N_BEQ_COR": "n_beq_cor",
    "OUTPUT.LHT": "lht",
    "OUTPUT.GAP": "gap",
}


def parse_file_metadata(filepath: str) -> dict[str, str]:
    """Infer architecture (monoplane/biplane), aero solver (AVL/LLT), and optimizer (PSO/patternsearch) from filename."""
    fname = os.path.basename(filepath)
    upper = fname.upper()

    if "MONO" in upper or fname == "individuos.xlsx":
        architecture = "monoplane"
    else:
        architecture = "biplane"

    if "AVL" in upper:
        solver = "AVL"
    else:
        solver = "LLT"

    if "PARTICLE" in upper:
        optimizer = "PSO"
    elif "PATTERN" in upper:
        optimizer = "PatternSearch"
    else:
        optimizer = "PSO"

    return {
        "source_file": fname,
        "architecture": architecture,
        "solver": solver,
        "optimizer": optimizer,
    }


def _vectorized_pv_and_pvoo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute component buildup Empty Weight (PV), Payload (CP), Efficiency (EE), and PVOO (Eq. 1.1) for all rows."""
    b_asa = df["b_asa"].to_numpy(dtype=float)
    cr_asa = df["cr_asa"].to_numpy(dtype=float)
    afil_asa = df["afil_asa"].to_numpy(dtype=float)
    pr_asa = df["pr_asa"].to_numpy(dtype=float)
    s_asa = df["s_asa"].to_numpy(dtype=float)

    b_sh = df["b_sh"].to_numpy(dtype=float)
    cr_sh = df["cr_sh"].to_numpy(dtype=float)
    afil_sh = df["afil_sh"].to_numpy(dtype=float)
    pr_sh = df["pr_sh"].to_numpy(dtype=float)
    s_sh = df["s_sh"].to_numpy(dtype=float)
    x_sh_ba = df["x_sh_ba"].to_numpy(dtype=float)
    is_biplane = (df["architecture"] == "biplane").to_numpy()

    # Fuselage (PV_FUSELAGEM.m)
    coefs_fus = np.polyfit([0.0, 1160.0, 1240.0, 1600.0], [0.0, 265.0, 305.0, 350.0], 2)
    massa_fus = np.polyval(coefs_fus, x_sh_ba * 1000.0) + 153.0 + 72.0

    # Spars, torsion box, covering, trailing edge, dowels
    coefs_long_asa = np.polyfit([0.0, 2.1, 2.5], [0.0, 75.0, 145.0], 1)
    coefs_long_sh = np.polyfit([0.0, 1.4, 2.1], [0.0, 33.0, 81.0], 1)
    massa_long_asa_mono = np.polyval(coefs_long_asa, b_asa)
    massa_long_asa_bip = b_asa * (70.0 / 2.2) * 2.0
    massa_long_sh = np.polyval(coefs_long_sh, b_sh)

    massa_ct_asa = (185.0 / 1.449) * s_asa
    massa_ct_sh = (57.0 / 0.35) * s_sh

    massa_entel_asa = 2.0 * s_asa * 40.0
    massa_entel_sh = 2.0 * s_sh * 40.0

    dens_balsa = 130.0
    esp_balsa = 1.5 / 1000.0
    massa_bf_asa = 0.1 * esp_balsa * b_asa * dens_balsa * 1000.0
    massa_bf_sh = 0.021 * esp_balsa * b_sh * dens_balsa * 1000.0
    massa_tar_asa = ((6.0 / 1000.0) ** 2) * b_asa * dens_balsa * 1000.0 * 2.0
    massa_tar_sh = ((4.0 / 1000.0) ** 2) * b_sh * dens_balsa * 1000.0

    # Smooth rib mass approximation calibrated to PV_NERVURAS.m
    cp_asa = cr_asa * afil_asa
    cp_sh = cr_sh * afil_sh
    c_mean_asa = pr_asa * cr_asa + (1.0 - pr_asa) * 0.5 * (cr_asa + cp_asa)
    c_mean_sh = pr_sh * cr_sh + (1.0 - pr_sh) * 0.5 * (cr_sh + cp_sh)
    n_ribs_asa = np.maximum(np.floor((b_asa * 100.0) / 11.5), 4.0)
    n_ribs_sh = np.maximum(np.floor((b_sh * 100.0) / 11.5), 3.0)
    massa_nerv_asa = (cr_asa * 100.0) * (85.0 / 68.0) + n_ribs_asa * (0.00117 * ((c_mean_asa * 100.0) ** 2))
    massa_nerv_sh = (cr_sh * 100.0) * (25.0 / 35.0) + n_ribs_sh * (0.00148 * ((c_mean_sh * 100.0) ** 2))

    coefs_prof = np.polyfit([0.0, 0.35, 0.63], [0.0, 29.0, 43.0], 2)
    massa_prof = np.polyval(coefs_prof, s_sh)
    coefs_ail = np.polyfit([0.0, 1.09, 1.449], [0.0, 26.0, 41.60], 1)

    massa_asa_mono = (
        massa_long_asa_mono + massa_ct_asa + massa_nerv_asa + massa_entel_asa + massa_bf_asa + massa_tar_asa
    ) * 1.2102 + np.where(b_asa > 3.0, 100.0, 0.0)
    massa_asa_bip = (
        massa_long_asa_bip
        + 2.0 * (massa_ct_asa + massa_nerv_asa + massa_entel_asa + massa_bf_asa + massa_tar_asa)
    ) * 1.4411 + np.where(b_asa > 3.0, 200.0, 0.0)

    massa_asa = np.where(is_biplane, massa_asa_bip, massa_asa_mono)
    massa_ail = np.polyval(coefs_ail, np.where(is_biplane, 2.0 * s_asa, s_asa))
    massa_sh = (massa_long_sh + massa_ct_sh + massa_nerv_sh + massa_entel_sh + massa_bf_sh + massa_tar_sh) * 1.11

    fixed_g = (
        255.0 + 231.0 + 40.0 + 250.0 + 100.0 + 39.0 + 83.0 + 46.0 + 15.0 + 123.2 + 15.0 + 8.0 + 16.0 + 9.0 + 2.5 + 46.3 + 58.0
    ) + (10.0 * x_sh_ba)

    pv = (massa_asa + massa_sh + massa_prof + massa_ail + massa_fus + fixed_g) / 1000.0
    mtow = df["mtow"].to_numpy(dtype=float)
    cp = np.maximum(mtow - pv, 0.0)
    ee = cp / np.maximum(pv, 1e-6)

    n_surf = np.where(is_biplane, 2.0, 1.0)
    f_n = 0.185 * (n_surf**2) - 0.775 * n_surf + 1.81
    p_voo = 50.0 * ee * f_n * (1.15 ** (-b_asa))

    df["pv"] = np.round(pv, 4)
    df["cp"] = np.round(cp, 4)
    df["ee"] = np.round(ee, 4)
    df["p_voo"] = np.round(p_voo, 4)

    # Derive normalized x_cg fraction from x_cg_m and MAC
    tr = afil_asa
    s_reta = pr_asa * b_asa * cr_asa
    s_afil = (1.0 - pr_asa) * 0.5 * (cr_asa + cp_asa) * b_asa
    s_tot = np.maximum(s_reta + s_afil, 1e-6)
    mac_afil = (2.0 / 3.0) * cr_asa * (1.0 + tr + tr**2) / np.maximum(1.0 + tr, 1e-6)
    mac = (s_afil * mac_afil + s_reta * cr_asa) / s_tot
    df["mac_asa"] = np.round(mac, 4)
    df["x_cg"] = np.round(np.clip(df["x_cg_m"].to_numpy(dtype=float) / np.maximum(mac, 1e-6), 0.05, 0.95), 4)

    # Feasibility flag according to Table 4.4 & MDO__FUNCAO_OBJETIVO.m
    ar_sh_col = df["out_ar_sh"].fillna(df["ar_sh"]) if "ar_sh" in df else df["out_ar_sh"]
    df["ar_sh"] = ar_sh_col
    feasible = (
        (df["mtow"] > df["pv"] + 0.5)
        & (df["ar_asa"] >= 4.09)
        & (df["ar_sh"] >= 3.95)
        & (df["cm_0"] >= 0.0)
        & (df["cm_a"] < 0.0)
        & (df["me"] >= 0.015)
        & (df["b_asa"] <= 4.0)
        & (df["x_dec"] <= 55.0)
    )
    df["is_feasible"] = feasible.astype(int)
    return df


def load_all_spreadsheets(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Read and unify all 20 Excel spreadsheets in TESTES_MDO/ and individuos.xlsx."""
    excel_files = sorted(glob.glob(str(raw_dir / "TESTES_MDO" / "*.xlsx")))
    ind_path = raw_dir / "individuos.xlsx"
    if ind_path.exists():
        excel_files.append(str(ind_path))

    frames: list[pd.DataFrame] = []
    for fpath in excel_files:
        meta = parse_file_metadata(fpath)
        raw_df = pd.read_excel(fpath)
        renamed = {}
        for col in raw_df.columns:
            col_str = str(col)
            if col_str in COLUMN_ALIASES:
                renamed[col] = COLUMN_ALIASES[col_str]
            elif col_str.strip() in COLUMN_ALIASES:
                renamed[col] = COLUMN_ALIASES[col_str.strip()]
        df = raw_df.rename(columns=renamed).copy()

        # Drop repeated header rows embedded inside spreadsheets
        df["mtow"] = pd.to_numeric(df["mtow"], errors="coerce")
        df = df.dropna(subset=["mtow", "b_asa", "cr_asa"]).copy()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "z_asa2" not in df.columns:
            df["z_asa2"] = np.nan
        if "gap" not in df.columns:
            df["gap"] = np.where(df["z_asa2"].notna(), df["z_asa2"] - df["z_asa"], 0.0)
        if "h_total" not in df.columns:
            df["h_total"] = np.maximum(df["z_asa"].fillna(0.2), df["z_sh"].fillna(0.4))
        if "ar_sh" not in df.columns and "out_ar_sh" in df.columns:
            df["ar_sh"] = df["out_ar_sh"]

        df["source_file"] = meta["source_file"]
        df["architecture"] = meta["architecture"]
        df["solver"] = meta["solver"]
        df["optimizer"] = meta["optimizer"]
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.dropna(
        subset=[
            "mtow",
            "b_asa",
            "z_asa",
            "afil_asa",
            "cr_asa",
            "pr_asa",
            "inc_asa",
            "b_sh",
            "x_sh_ba",
            "z_sh",
            "afil_sh",
            "cr_sh",
            "pr_sh",
            "inc_sh",
            "cm_0",
            "cm_a",
            "me",
            "x_dec",
            "cl_max_voo",
        ]
    ).reset_index(drop=True)

    full_df = _vectorized_pv_and_pvoo(full_df)

    ordered_cols = [
        "source_file",
        "architecture",
        "solver",
        "optimizer",
        "is_feasible",
        "mtow",
        "pv",
        "cp",
        "ee",
        "p_voo",
        "objective_f",
        "penalty",
        "x_cg",
        "b_asa",
        "z_asa",
        "z_asa2",
        "gap",
        "afil_asa",
        "cr_asa",
        "pr_asa",
        "inc_asa",
        "b_sh",
        "x_sh_ba",
        "z_sh",
        "afil_sh",
        "cr_sh",
        "pr_sh",
        "inc_sh",
        "ar_asa",
        "ar_sh",
        "cm_0",
        "cm_a",
        "me",
        "h_total",
        "v_dec",
        "x_dec",
        "cl_max_solo",
        "cl_max_voo",
        "estol_av_solo",
        "estol_av_voo",
        "s_asa",
        "s_sh",
        "mac_asa",
        "lht",
    ]
    existing_cols = [c for c in ordered_cols if c in full_df.columns]
    return full_df[existing_cols]


def extract_and_chunk_pdf_report(
    pdf_path: Path = DEFAULT_RAW_DIR / "Relatorio_de_Projeto_Conceitual_Preliminar_e_Integracao_2026.pdf",
) -> list[dict[str, Any]]:
    """Extract text per page from the 2026 Report PDF and split into section-and-requirement-indexed chunks."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Report PDF not found at {pdf_path}")

    raw_text = subprocess.check_output(["pdftotext", "-layout", str(pdf_path), "-"], text=True)
    pages = raw_text.split("\x0c")

    section_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-ZÀ-Ú][^\n\.]{2,70})$", re.MULTILINE)
    req_pattern = re.compile(r"REQ-[A-Z]+-\d+|DG-\d+")
    footer_page_pattern = re.compile(r"Página:\s*(\d+)\s*de\s*\d+")

    current_section = "1 Introdução"
    chunks: list[dict[str, Any]] = []

    for page_idx, page_text in enumerate(pages, start=1):
        footer_match = footer_page_pattern.search(page_text)
        printed_page = int(footer_match.group(1)) if footer_match else page_idx

        cleaned_lines: list[str] = []
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Página:") or ". . . ." in stripped:
                continue
            cleaned_lines.append(stripped)

        if not cleaned_lines:
            continue

        joined_page = "\n".join(cleaned_lines)
        sec_matches = list(section_pattern.finditer(joined_page))
        if sec_matches:
            last_m = sec_matches[-1]
            current_section = f"{last_m.group(1)} {last_m.group(2).strip()}"

        # Split page into semantic paragraphs (~450-800 chars) with overlap
        paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?=\b\d+\.\d+(?:\.\d+)?\s+[A-ZÀ-Ú])", joined_page) if len(p.strip()) > 35]
        if not paragraphs:
            paragraphs = [joined_page]

        buffer = ""
        sub_idx = 1
        for para in paragraphs:
            if len(buffer) + len(para) <= 750:
                buffer = f"{buffer}\n{para}".strip()
            else:
                if len(buffer) >= 45:
                    reqs = sorted(set(req_pattern.findall(buffer)))
                    chunks.append(
                        {
                            "chunk_id": f"p{printed_page:02d}_c{sub_idx:02d}",
                            "page": printed_page,
                            "section": current_section,
                            "requirement_ids": json.dumps(reqs, ensure_ascii=False),
                            "content": buffer,
                        }
                    )
                    sub_idx += 1
                buffer = para

        if len(buffer) >= 45:
            reqs = sorted(set(req_pattern.findall(buffer)))
            chunks.append(
                {
                    "chunk_id": f"p{printed_page:02d}_c{sub_idx:02d}",
                    "page": printed_page,
                    "section": current_section,
                    "requirement_ids": json.dumps(reqs, ensure_ascii=False),
                    "content": buffer,
                }
            )

    return chunks


def build_database(
    db_path: Path = DEFAULT_DB_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> dict[str, Any]:
    """Build SQLite database with mdo_evaluations and doc_chunks tables + indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_all_spreadsheets(raw_dir=raw_dir)
    chunks = extract_and_chunk_report_safe(raw_dir=raw_dir)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("mdo_evaluations", conn, if_exists="replace", index=False)
        chunks_df = pd.DataFrame(chunks)
        chunks_df.to_sql("doc_chunks", conn, if_exists="replace", index=False)

        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_arch_solver ON mdo_evaluations(architecture, solver, is_feasible)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_pvoo ON mdo_evaluations(p_voo DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_mtow ON mdo_evaluations(mtow DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_page ON doc_chunks(page)")
        conn.commit()

    summary = {
        "total_evaluations": int(len(df)),
        "feasible_evaluations": int(df["is_feasible"].sum()),
        "by_architecture": {str(k): int(v) for k, v in df["architecture"].value_counts().items()},
        "by_solver": {str(k): int(v) for k, v in df["solver"].value_counts().items()},
        "by_optimizer": {str(k): int(v) for k, v in df["optimizer"].value_counts().items()},
        "total_doc_chunks": len(chunks),
        "db_path": str(db_path),
    }
    return summary


def extract_and_chunk_report_safe(raw_dir: Path = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    """Extract PDF report chunks directly from Relatório 2026 without synthetic chunks."""
    pdf_path = raw_dir / "Relatorio_de_Projeto_Conceitual_Preliminar_e_Integracao_2026.pdf"
    return extract_and_chunk_pdf_report(pdf_path)
