"""Aerodynamic & Multidisciplinary Surrogate Modeling (XGBoost / RF / Poly-Ridge) and Multi-Fidelity PSO.

Reproduces and audits:
- Section 4.1.4 & Table 4.3 (Relatório 2026): Surrogate validation with R², RMSE, and MAPE (%) on both
  real AVL solver campaigns (`solver == 'AVL'`) and LLT aerodynamic polars.
- Leak-free evaluation:
  1. Strictly raw geometric design variables (`RAW_FEATURE_COLS`, 16 inputs) — derived constraint quantities
     (`ar_asa`, `ar_sh`, `s_asa`, `s_sh`, `lht`) are excluded from model inputs to prevent feature leakage.
  2. Both 5-fold `GroupKFold` grouped by `source_file` (zero campaign overlap) and deduplicated 80/20 split.
- Section 5.4.1 (Relatório 2026): Multi-seed Particle Swarm Optimization (PSO) benchmark across 5 random seeds,
  comparing Full-Physics PSO (200 calls), Multi-Fidelity Surrogate-Assisted PSO (40 calls), and Equal-Budget
  Full-Physics PSO (40 calls).
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from aeromdo.data_pipeline import DEFAULT_DB_PATH
from aeromdo.physics import (
    LiftingSurface,
    compute_geometry,
    get_airfoil_profile,
    run_llt,
    run_simulation,
    sample_doe_lhs,
)
from aeromdo.schemas import AircraftInput, SurrogatePrediction


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

# Strictly raw design variables (zero derived constraint variables to prevent leakage)
FEATURE_COLS: list[str] = [
    "is_biplane",
    "x_cg",
    "b_asa",
    "z_asa",
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
]

REGRESSION_TARGETS: list[str] = [
    "cl_max_voo",
    "cl_max_solo",
    "mtow",
    "pv",
    "p_voo",
    "x_dec",
    "cm_0",
    "cm_a",
    "me",
]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract strictly raw geometric input parameters (no derived constraint features)."""
    out = pd.DataFrame(index=df.index)
    out["is_biplane"] = (df["architecture"] == "biplane").astype(float)
    out["x_cg"] = df["x_cg"].astype(float)
    out["b_asa"] = df["b_asa"].astype(float)
    out["z_asa"] = df["z_asa"].astype(float)
    out["gap"] = df["gap"].fillna(0.0).astype(float)
    out["afil_asa"] = df["afil_asa"].astype(float)
    out["cr_asa"] = df["cr_asa"].astype(float)
    out["pr_asa"] = df["pr_asa"].astype(float)
    out["inc_asa"] = df["inc_asa"].astype(float)
    out["b_sh"] = df["b_sh"].astype(float)
    out["x_sh_ba"] = df["x_sh_ba"].astype(float)
    out["z_sh"] = df["z_sh"].astype(float)
    out["afil_sh"] = df["afil_sh"].astype(float)
    out["cr_sh"] = df["cr_sh"].astype(float)
    out["pr_sh"] = df["pr_sh"].astype(float)
    out["inc_sh"] = df["inc_sh"].astype(float)
    return out[FEATURE_COLS]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R², RMSE, and MAPE (%) as in Table 4.3 of Relatório 2026."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    # Use a threshold of 0.05 to avoid division-by-zero blowup on near-zero stability derivatives (CM0, CMA, ME)
    mask = np.abs(y_true) > 0.05
    if np.any(mask):
        mape_pct = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0)
    else:
        mape_pct = 0.0
    return {
        "r2": round(r2, 5),
        "rmse": round(rmse, 5),
        "mape_pct": round(mape_pct, 3),
    }


def evaluate_real_avl_subset(df: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Evaluate XGBoost surrogates strictly on the real AVL VLM solver subset (`solver == 'AVL'`)."""
    avl_df = df[
        (df["solver"] == "AVL")
        & (df["mtow"] >= 5.0)
        & (df["mtow"] <= 20.5)
        & (df["cl_max_voo"] > 0.3)
        & (df["cl_max_voo"] < 4.5)
        & (df["x_dec"] > 5.0)
        & (df["x_dec"] <= 65.0)
        & (df["me"] > -0.5)
        & (df["me"] < 0.8)
    ].copy()

    dedup_cols = ["architecture", "b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba"]
    rounded = avl_df[dedup_cols].copy()
    for col in ["b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba"]:
        rounded[col] = rounded[col].round(4)
    avl_dedup = avl_df.loc[~rounded.duplicated()].reset_index(drop=True)

    X_avl = _prepare_features(avl_dedup)
    idx_tr, idx_te = train_test_split(
        np.arange(len(avl_dedup)),
        test_size=0.20,
        random_state=seed,
    )

    avl_targets = ["cl_max_voo", "cl_max_solo", "cm_0", "cm_a", "me", "mtow", "x_dec"]
    metrics: dict[str, dict[str, float]] = {}
    for target in avl_targets:
        y = avl_dedup[target].to_numpy(dtype=float)
        reg = XGBRegressor(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
        )
        reg.fit(X_avl.iloc[idx_tr], y[idx_tr])
        pred = reg.predict(X_avl.iloc[idx_te])
        metrics[target] = compute_regression_metrics(y[idx_te], pred)

    return {
        "total_avl_rows": int(len(avl_df)),
        "unique_avl_rows": int(len(avl_dedup)),
        "test_rows": int(len(idx_te)),
        "metrics_by_target": metrics,
    }


def evaluate_aerodynamic_polar_surrogate(seed: int = 42) -> dict[str, Any]:
    """Reproduce Table 4.3 (Relatório 2026) on polar curves CL(alpha) and CD(CL) with grouped geometry split."""
    samples = sample_doe_lhs(n_samples=600, architecture="monoplane", seed=seed)
    rows: list[dict[str, float]] = []
    for geom_id, s in enumerate(samples):
        asa = LiftingSurface(
            name="ASA",
            b=s.b_asa,
            cr=s.cr_asa,
            cp=s.cr_asa * s.afil_asa,
            pr=s.pr_asa,
            inc=s.inc_asa,
            pos_ba=np.array([0.0, 0.0, s.z_asa]),
            pos=np.array([0.15, 0.0, s.z_asa]),
            perfil=get_airfoil_profile(1),
        )
        compute_geometry(asa)
        _, voo = run_llt(asa)
        for alpha in (-4.0, 0.0, 4.0, 8.0, 12.0):
            cl_val = float(np.polyval(voo.curva_cl, alpha))
            cd_val = float(np.polyval(voo.curva_cd[13], alpha))
            rows.append(
                {
                    "geom_id": float(geom_id),
                    "b_asa": asa.b,
                    "cr_asa": asa.cr,
                    "afil_asa": asa.afil,
                    "pr_asa": asa.pr,
                    "alpha": alpha,
                    "cl": cl_val,
                    "cd": cd_val,
                }
            )
    polar_df = pd.DataFrame(rows)
    # Split by geom_id so all 5 angles of a test wing geometry are unseen during training
    unique_geoms = np.arange(len(samples))
    tr_geoms, te_geoms = train_test_split(unique_geoms, test_size=0.2, random_state=seed)
    tr_mask = polar_df["geom_id"].isin(tr_geoms).to_numpy()
    te_mask = polar_df["geom_id"].isin(te_geoms).to_numpy()

    X_cl = polar_df[["b_asa", "cr_asa", "afil_asa", "pr_asa", "alpha"]]
    y_cl = polar_df["cl"].to_numpy()

    X_cd = polar_df[["b_asa", "cr_asa", "afil_asa", "pr_asa", "alpha", "cl"]]
    y_cd = polar_df["cd"].to_numpy()

    model_cl = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, random_state=seed, n_jobs=4)
    model_cl.fit(X_cl.iloc[tr_mask], y_cl[tr_mask])
    pred_cl = model_cl.predict(X_cl.iloc[te_mask])

    model_cd = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, random_state=seed, n_jobs=4)
    model_cd.fit(X_cd.iloc[tr_mask], y_cd[tr_mask])
    pred_cd = model_cd.predict(X_cd.iloc[te_mask])

    cl_metrics = compute_regression_metrics(y_cl[te_mask], pred_cl)
    cd_metrics = compute_regression_metrics(y_cd[te_mask], pred_cd)

    return {
        "split_protocol": "grouped_by_wing_geometry_id (480 train wings, 120 unseen test wings)",
        "cl_alpha": {
            **cl_metrics,
            "target_r2": 0.992,
            "target_rmse": 0.012,
            "target_mape_pct": 1.2,
            "passes_report_table_4_3": (
                cl_metrics["r2"] > 0.992 and cl_metrics["rmse"] < 0.012 and cl_metrics["mape_pct"] < 1.2
            ),
        },
        "cd_cl": {
            **cd_metrics,
            "target_r2": 0.985,
            "target_rmse": 0.003,
            "target_mape_pct": 2.1,
            "passes_report_table_4_3": (
                cd_metrics["r2"] > 0.985 and cd_metrics["rmse"] < 0.003 and cd_metrics["mape_pct"] < 2.1
            ),
        },
    }


class MDOSurrogateSuite:
    """Trained ensemble of XGBoost regressors & leak-free feasibility classifier for aircraft MDO."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.regressors: dict[str, XGBRegressor] = {}
        self.feasibility_clf: XGBClassifier | None = None
        self.metrics_summary: dict[str, Any] = {}

    def train_and_evaluate(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Train surrogate models on historical AVL + LLT evaluations with both Random and GroupKFold splits."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM mdo_evaluations", conn)

        # Clean extreme outliers from unstable penalty explosions
        valid_df = df[
            (df["mtow"] >= 5.0)
            & (df["mtow"] <= 20.5)
            & (df["cl_max_voo"] > 0.3)
            & (df["cl_max_voo"] < 4.5)
            & (df["x_dec"] > 5.0)
            & (df["x_dec"] <= 65.0)
            & (df["me"] > -0.5)
            & (df["me"] < 0.8)
        ].copy()

        # Deduplicate near-identical PSO convergence particles
        dedup_cols = ["architecture", "solver", "b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba"]
        rounded = valid_df[dedup_cols].copy()
        for col in ["b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba"]:
            rounded[col] = rounded[col].round(4)
        valid_df = valid_df.loc[~rounded.duplicated()].reset_index(drop=True)

        X = _prepare_features(valid_df)
        y_feas = valid_df["is_feasible"].to_numpy(dtype=int)
        groups = valid_df["source_file"].to_numpy()

        # ------------------------------------------------------------------
        # Protocol A: 5-Fold GroupKFold by `source_file` (zero campaign leakage)
        # ------------------------------------------------------------------
        gkf = GroupKFold(n_splits=5)
        oof_prob_feas = np.zeros(len(valid_df), dtype=float)
        oof_preds_reg: dict[str, np.ndarray] = {
            t: np.zeros(len(valid_df), dtype=float) for t in ("cl_max_voo", "mtow", "p_voo", "x_dec", "cm_0", "cm_a", "me")
        }

        for tr_idx, te_idx in gkf.split(X, y_feas, groups=groups):
            clf_fold = XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=4,
                eval_metric="logloss",
            )
            clf_fold.fit(X.iloc[tr_idx], y_feas[tr_idx])
            oof_prob_feas[te_idx] = clf_fold.predict_proba(X.iloc[te_idx])[:, 1]

            for t in oof_preds_reg:
                y_target = valid_df[t].to_numpy(dtype=float)
                reg_fold = XGBRegressor(
                    n_estimators=180,
                    max_depth=5,
                    learning_rate=0.06,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    random_state=seed,
                    n_jobs=4,
                )
                reg_fold.fit(X.iloc[tr_idx], y_target[tr_idx])
                oof_preds_reg[t][te_idx] = reg_fold.predict(X.iloc[te_idx])

        oof_pred_feas = (oof_prob_feas >= 0.5).astype(int)
        grouped_cv_feas_metrics = {
            "roc_auc": round(float(roc_auc_score(y_feas, oof_prob_feas)), 4),
            "f1": round(float(f1_score(y_feas, oof_pred_feas)), 4),
            "precision": round(float(precision_score(y_feas, oof_pred_feas)), 4),
            "recall": round(float(recall_score(y_feas, oof_pred_feas)), 4),
            "n_campaigns": int(valid_df["source_file"].nunique()),
        }
        grouped_cv_reg_metrics = {
            t: compute_regression_metrics(valid_df[t].to_numpy(dtype=float), pred_arr)
            for t, pred_arr in oof_preds_reg.items()
        }

        # ------------------------------------------------------------------
        # Protocol B: Deduplicated 80/20 Stratified Split (within-domain interpolation)
        # ------------------------------------------------------------------
        idx_tr, idx_te = train_test_split(
            np.arange(len(valid_df)),
            test_size=0.20,
            random_state=seed,
            stratify=y_feas,
        )
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]

        # 1. Feasibility Classifier (XGBoost on strictly raw features)
        clf = XGBClassifier(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
            eval_metric="logloss",
        )
        clf.fit(X_tr, y_feas[idx_tr])
        prob_feas = clf.predict_proba(X_te)[:, 1]
        pred_feas = (prob_feas >= 0.5).astype(int)

        feas_metrics = {
            "roc_auc": round(float(roc_auc_score(y_feas[idx_te], prob_feas)), 4),
            "f1": round(float(f1_score(y_feas[idx_te], pred_feas)), 4),
            "precision": round(float(precision_score(y_feas[idx_te], pred_feas)), 4),
            "recall": round(float(recall_score(y_feas[idx_te], pred_feas)), 4),
            "test_samples": int(len(idx_te)),
            "features_used": FEATURE_COLS,
        }
        self.feasibility_clf = clf

        # 2. Multi-target Regressors + Baseline Comparison on p_voo, cl_max_voo, mtow
        reg_metrics: dict[str, dict[str, float]] = {}
        baseline_comparison: dict[str, dict[str, dict[str, float]]] = {}

        for target in REGRESSION_TARGETS:
            y_all = valid_df[target].to_numpy(dtype=float)
            y_tr, y_te = y_all[idx_tr], y_all[idx_te]

            xgb_reg = XGBRegressor(
                n_estimators=240,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=4,
            )
            xgb_reg.fit(X_tr, y_tr)
            pred_xgb = xgb_reg.predict(X_te)
            reg_metrics[target] = compute_regression_metrics(y_te, pred_xgb)
            self.regressors[target] = xgb_reg

            if target in ("p_voo", "cl_max_voo", "mtow"):
                poly_ridge = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=2, include_bias=False),
                    Ridge(alpha=1.0),
                )
                poly_ridge.fit(X_tr, y_tr)
                pred_poly = poly_ridge.predict(X_te)

                rf_reg = RandomForestRegressor(
                    n_estimators=80, max_depth=12, random_state=seed, n_jobs=4
                )
                rf_reg.fit(X_tr, y_tr)
                pred_rf = rf_reg.predict(X_te)

                baseline_comparison[target] = {
                    "Polynomial_Ridge_Deg2": compute_regression_metrics(y_te, pred_poly),
                    "RandomForest": compute_regression_metrics(y_te, pred_rf),
                    "XGBoost_Surrogate": reg_metrics[target],
                }

        # 3. Real AVL VLM subset validation (6,576 rows from AVL spreadsheets)
        avl_validation = evaluate_real_avl_subset(df, seed=seed)

        # 4. Polar validation reproducing Table 4.3 of Relatório 2026 (grouped by wing geometry)
        polar_table_4_3 = evaluate_aerodynamic_polar_surrogate(seed=seed)

        self.metrics_summary = {
            "dataset_total_rows": int(len(df)),
            "dataset_unique_cleaned_rows": int(len(valid_df)),
            "train_rows": int(len(idx_tr)),
            "test_rows": int(len(idx_te)),
            "raw_feature_columns": FEATURE_COLS,
            "table_4_3_polar_validation": polar_table_4_3,
            "real_avl_solver_validation": avl_validation,
            "mdo_regression_metrics": reg_metrics,
            "feasibility_classifier_metrics": feas_metrics,
            "grouped_by_source_file_cv": {
                "feasibility_classifier": grouped_cv_feas_metrics,
                "regression": grouped_cv_reg_metrics,
            },
            "model_family_benchmark": baseline_comparison,
        }

        self.save()
        return self.metrics_summary

    def save(self) -> None:
        """Persist trained models and metrics to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "regressors": self.regressors,
            "feasibility_clf": self.feasibility_clf,
            "metrics_summary": self.metrics_summary,
        }
        with open(self.model_dir / "surrogate_bundle.pkl", "wb") as f:
            pickle.dump(bundle, f)
        with open(self.model_dir / "surrogate_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        """Load trained models from disk if available."""
        bundle_path = self.model_dir / "surrogate_bundle.pkl"
        if not bundle_path.exists():
            return False
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        self.regressors = bundle["regressors"]
        self.feasibility_clf = bundle["feasibility_clf"]
        self.metrics_summary = bundle.get("metrics_summary", {})
        return True

    def predict(self, inp: AircraftInput) -> SurrogatePrediction:
        """Run millisecond surrogate inference for a single AircraftInput."""
        if not self.regressors or self.feasibility_clf is None:
            if not self.load():
                raise RuntimeError("Surrogate models are not trained yet. Run train_and_evaluate() first.")

        t0 = time.perf_counter()
        gap = (
            max((inp.z_asa2 or (inp.z_asa + 0.25)) - inp.z_asa, 0.10)
            if inp.architecture == "biplane"
            else 0.0
        )
        row_df = pd.DataFrame(
            [
                {
                    "architecture": inp.architecture,
                    "x_cg": inp.x_cg,
                    "b_asa": inp.b_asa,
                    "z_asa": inp.z_asa,
                    "gap": gap,
                    "afil_asa": inp.afil_asa,
                    "cr_asa": inp.cr_asa,
                    "pr_asa": inp.pr_asa,
                    "inc_asa": inp.inc_asa,
                    "b_sh": inp.b_sh,
                    "x_sh_ba": inp.x_sh_ba,
                    "z_sh": inp.z_sh,
                    "afil_sh": inp.afil_sh,
                    "cr_sh": inp.cr_sh,
                    "pr_sh": inp.pr_sh,
                    "inc_sh": inp.inc_sh,
                }
            ]
        )
        X_feat = _prepare_features(row_df)

        preds = {target: float(model.predict(X_feat)[0]) for target, model in self.regressors.items()}
        prob_feas = float(self.feasibility_clf.predict_proba(X_feat)[0, 1])

        cp_asa = inp.cr_asa * inp.afil_asa
        s_asa = inp.pr_asa * inp.b_asa * inp.cr_asa + (1.0 - inp.pr_asa) * 0.5 * (inp.cr_asa + cp_asa) * inp.b_asa
        ar_asa = (inp.b_asa**2) / max(s_asa, 1e-6)

        cp_sh = inp.cr_sh * inp.afil_sh
        s_sh = inp.pr_sh * inp.b_sh * inp.cr_sh + (1.0 - inp.pr_sh) * 0.5 * (inp.cr_sh + cp_sh) * inp.b_sh
        ar_sh = (inp.b_sh**2) / max(s_sh, 1e-6)

        geom_ok = (
            ar_asa >= 4.1
            and ar_sh >= 4.1
            and inp.b_asa <= 4.0
            and preds["cm_0"] >= 0.0
            and preds["cm_a"] < 0.0
            and preds["me"] >= 0.015
            and preds["x_dec"] <= 55.0
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SurrogatePrediction(
            architecture=inp.architecture,
            solver_target="AVL",
            predicted_mtow=round(preds["mtow"], 4),
            predicted_pv=round(preds["pv"], 4),
            predicted_p_voo=round(preds["p_voo"], 4),
            predicted_cl_max_voo=round(preds["cl_max_voo"], 4),
            predicted_x_dec=round(preds["x_dec"], 3),
            predicted_cm_0=round(preds["cm_0"], 5),
            predicted_cm_a=round(preds["cm_a"], 5),
            predicted_me=round(preds["me"], 5),
            feasibility_probability=round(prob_feas, 4),
            is_likely_feasible=(prob_feas >= 0.5 and geom_ok),
            inference_time_ms=round(elapsed_ms, 3),
        )

    def predict_batch_df(self, candidates: list[AircraftInput]) -> pd.DataFrame:
        """Vectorized batch inference across candidate geometries in a single XGBoost call."""
        if not self.regressors or self.feasibility_clf is None:
            if not self.load():
                raise RuntimeError("Surrogate models are not trained yet.")

        records = []
        for inp in candidates:
            gap = (
                max((inp.z_asa2 or (inp.z_asa + 0.25)) - inp.z_asa, 0.10)
                if inp.architecture == "biplane"
                else 0.0
            )
            records.append(
                {
                    "architecture": inp.architecture,
                    "x_cg": inp.x_cg,
                    "b_asa": inp.b_asa,
                    "z_asa": inp.z_asa,
                    "gap": gap,
                    "afil_asa": inp.afil_asa,
                    "cr_asa": inp.cr_asa,
                    "pr_asa": inp.pr_asa,
                    "inc_asa": inp.inc_asa,
                    "b_sh": inp.b_sh,
                    "x_sh_ba": inp.x_sh_ba,
                    "z_sh": inp.z_sh,
                    "afil_sh": inp.afil_sh,
                    "cr_sh": inp.cr_sh,
                    "pr_sh": inp.pr_sh,
                    "inc_sh": inp.inc_sh,
                }
            )
        raw_df = pd.DataFrame(records)
        X_feat = _prepare_features(raw_df)
        out = X_feat.copy()
        for target, model in self.regressors.items():
            out[f"pred_{target}"] = model.predict(X_feat)
        out["feasibility_prob"] = self.feasibility_clf.predict_proba(X_feat)[:, 1]

        # Compute exact AR_ASA and AR_SH for acquisition penalty calculation
        cp_asa = out["cr_asa"] * out["afil_asa"]
        s_asa = out["pr_asa"] * out["b_asa"] * out["cr_asa"] + (1.0 - out["pr_asa"]) * 0.5 * (out["cr_asa"] + cp_asa) * out["b_asa"]
        out["ar_asa"] = (out["b_asa"] ** 2) / np.maximum(s_asa, 1e-6)

        cp_sh = out["cr_sh"] * out["afil_sh"]
        s_sh = out["pr_sh"] * out["b_sh"] * out["cr_sh"] + (1.0 - out["pr_sh"]) * 0.5 * (out["cr_sh"] + cp_sh) * out["b_sh"]
        out["ar_sh"] = (out["b_sh"] ** 2) / np.maximum(s_sh, 1e-6)
        return out


def run_multifidelity_pso_benchmark(
    surrogate: MDOSurrogateSuite,
    n_particles: int = 20,
    n_iterations: int = 10,
    top_k_high_fidelity: int = 4,
    architecture: str = "monoplane",
    seed: int = 42,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Multi-seed benchmark comparing Full-Physics PSO, Multi-Fidelity Surrogate PSO, and Equal-Budget PSO.

    Evaluates across 5 independent random seeds (`[42, 101, 202, 303, 404]` by default):
    1. Standard Full-Physics PSO: `n_particles` × `n_iterations` = 200 physics calls/run
    2. Multi-Fidelity Surrogate-Assisted PSO: `n_particles` × `n_iterations` screened by surrogate,
       only `top_k_high_fidelity` verified by physics = 40 physics calls/run (80% call reduction)
    3. Equal-Budget Full-Physics PSO: 8 particles × 5 iterations = 40 physics calls/run (no surrogate)
    """
    seed_list = seeds if seeds is not None else [seed, 101, 202, 303, 404]

    bounds = np.array(
        [
            [0.20, 0.45],  # x_cg
            [2.00, 3.20],  # b_asa
            [0.16, 0.28],  # z_asa
            [0.52, 0.68],  # afil_asa
            [0.45, 0.68],  # cr_asa
            [0.35, 0.65],  # pr_asa
            [-1.0, 2.0],   # inc_asa
            [0.85, 1.40],  # b_sh
            [1.35, 1.85],  # x_sh_ba
            [0.25, 0.50],  # z_sh
            [0.55, 0.85],  # afil_sh
            [0.18, 0.30],  # cr_sh
            [0.40, 0.75],  # pr_sh
            [-3.0, 0.0],   # inc_sh
        ]
    )
    dim = len(bounds)

    def vec_to_input(x: np.ndarray) -> AircraftInput:
        z_asa = float(x[2])
        return AircraftInput(
            architecture="biplane" if architecture == "biplane" else "monoplane",
            x_cg=round(float(x[0]), 4),
            b_asa=round(float(x[1]), 4),
            z_asa=round(z_asa, 4),
            z_asa2=round(z_asa + 0.28, 4) if architecture == "biplane" else None,
            afil_asa=round(float(x[3]), 4),
            cr_asa=round(float(x[4]), 4),
            pr_asa=round(float(x[5]), 4),
            inc_asa=float(round(x[6])),
            b_sh=round(float(x[7]), 4),
            x_sh_ba=round(float(x[8]), 4),
            z_sh=round(float(x[9]), 4),
            afil_sh=round(float(x[10]), 4),
            cr_sh=round(float(x[11]), 4),
            pr_sh=round(float(x[12]), 4),
            inc_sh=float(round(x[13])),
            gmp=1,
        )

    def run_single_pso(
        run_seed: int,
        particles: int,
        iterations: int,
        multifidelity: bool,
        top_k: int,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(run_seed)
        pos = rng.uniform(bounds[:, 0], bounds[:, 1], size=(particles, dim))
        vel = np.zeros_like(pos)
        pbest_pos = pos.copy()
        pbest_obj = np.full(particles, float("inf"))
        gbest_pos = pos[0].copy()
        gbest_obj = float("inf")
        gbest_result = None

        physics_calls = 0
        surrogate_calls = 0
        history_best_pvoo: list[float] = []
        archive_x: list[np.ndarray] = []
        archive_obj: list[float] = []
        t_start = time.perf_counter()

        w_inertia = 0.68
        c1 = 1.45
        c2 = 1.45

        for _ in range(iterations):
            candidates = [vec_to_input(pos[i]) for i in range(particles)]

            if not multifidelity:
                for i, cand in enumerate(candidates):
                    res = run_simulation(cand)
                    physics_calls += 1
                    obj = res.outputs.objective_f if res.constraints.is_feasible else (res.outputs.objective_f + 1e5)
                    if obj < pbest_obj[i]:
                        pbest_obj[i] = obj
                        pbest_pos[i] = pos[i].copy()
                    if obj < gbest_obj:
                        gbest_obj = obj
                        gbest_pos = pos[i].copy()
                        gbest_result = res
            else:
                batch_df = surrogate.predict_batch_df(candidates)
                surrogate_calls += particles

                ar_pen = np.maximum(0.0, 4.1 - batch_df["ar_asa"].to_numpy()) + np.maximum(0.0, 4.1 - batch_df["ar_sh"].to_numpy())
                stab_pen = (
                    np.maximum(0.0, -batch_df["pred_cm_0"].to_numpy())
                    + np.maximum(0.0, batch_df["pred_cm_a"].to_numpy())
                    + np.maximum(0.0, 0.02 - batch_df["pred_me"].to_numpy())
                )
                # Physics-informed PVOO using exact analytical Eq. 1.1 with surrogate MTOW & PV
                b_arr = batch_df["b_asa"].to_numpy()
                pv_arr = np.maximum(batch_df["pred_pv"].to_numpy(), 1.5)
                cp_arr = np.maximum(batch_df["pred_mtow"].to_numpy() - pv_arr, 0.0)
                f_n = 1.0 if architecture == "biplane" else 1.22
                phys_pvoo = 50.0 * (cp_arr / pv_arr) * f_n * (1.15 ** (-b_arr))

                surr_obj = (
                    -phys_pvoo * batch_df["feasibility_prob"].to_numpy()
                    + 5000.0 * (ar_pen + stab_pen)
                )

                # Online active-learning correction from high-fidelity archive collected during this PSO run
                if len(archive_x) >= 8:
                    local_rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=run_seed)
                    local_rf.fit(np.asarray(archive_x), np.asarray(archive_obj))
                    local_pred = local_rf.predict(pos)
                    surr_obj = 0.5 * surr_obj + 0.5 * local_pred

                top_indices = np.argsort(surr_obj)[:top_k]
                top_set = set(int(idx) for idx in top_indices)

                for i in range(particles):
                    if i in top_set:
                        res = run_simulation(candidates[i])
                        physics_calls += 1
                        obj = res.outputs.objective_f if res.constraints.is_feasible else (res.outputs.objective_f + 1e5)
                        archive_x.append(pos[i].copy())
                        archive_obj.append(float(res.outputs.objective_f if res.constraints.is_feasible else 0.0))
                        if obj < pbest_obj[i]:
                            pbest_obj[i] = obj
                            pbest_pos[i] = pos[i].copy()
                        if obj < gbest_obj:
                            gbest_obj = obj
                            gbest_pos = pos[i].copy()
                            gbest_result = res
                    else:
                        if pbest_obj[i] == float("inf") and ar_pen[i] == 0.0:
                            pbest_obj[i] = float(surr_obj[i]) + 50.0
                            pbest_pos[i] = pos[i].copy()

            best_pvoo_now = float(gbest_result.outputs.p_voo) if (gbest_result is not None and gbest_result.constraints.is_feasible) else 0.0
            history_best_pvoo.append(round(best_pvoo_now, 3))

            r1 = rng.random(size=(particles, dim))
            r2 = rng.random(size=(particles, dim))
            vel = w_inertia * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
            pos = np.clip(pos + vel, bounds[:, 0], bounds[:, 1])

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        is_feas = bool(gbest_result.constraints.is_feasible) if gbest_result else False
        return {
            "seed": run_seed,
            "best_p_voo": round(float(gbest_result.outputs.p_voo), 4) if (gbest_result and is_feas) else 0.0,
            "best_mtow": round(float(gbest_result.outputs.mtow), 4) if gbest_result else 0.0,
            "best_pv": round(float(gbest_result.outputs.pv), 4) if gbest_result else 0.0,
            "best_b_asa": round(float(gbest_result.inputs.b_asa), 4) if gbest_result else 0.0,
            "best_me": round(float(gbest_result.constraints.me), 4) if gbest_result else 0.0,
            "is_feasible": is_feas,
            "physics_solver_calls": int(physics_calls),
            "surrogate_calls": int(surrogate_calls),
            "wall_time_ms": round(elapsed_ms, 2),
            "convergence_history": history_best_pvoo,
        }

    # Equal-budget parameters (same total physics calls = top_k_high_fidelity * n_iterations)
    total_mf_calls = top_k_high_fidelity * n_iterations
    eq_iters = max(n_iterations // 2, 2)
    eq_particles = max(total_mf_calls // eq_iters, 2)

    std_runs = [run_single_pso(s, n_particles, n_iterations, False, n_particles) for s in seed_list]
    mf_runs = [run_single_pso(s, n_particles, n_iterations, True, top_k_high_fidelity) for s in seed_list]
    eq_runs = [run_single_pso(s, eq_particles, eq_iters, False, eq_particles) for s in seed_list]

    def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
        pvoos = np.array([r["best_p_voo"] for r in runs], dtype=float)
        times = np.array([r["wall_time_ms"] for r in runs], dtype=float)
        calls = int(np.mean([r["physics_solver_calls"] for r in runs]))
        surr = int(np.mean([r["surrogate_calls"] for r in runs]))
        best_idx = int(np.argmax(pvoos))
        return {
            "best_p_voo": round(float(np.mean(pvoos)), 4),
            "std_p_voo": round(float(np.std(pvoos)), 4),
            "min_p_voo": round(float(np.min(pvoos)), 4),
            "max_p_voo": round(float(np.max(pvoos)), 4),
            "best_mtow": runs[best_idx]["best_mtow"],
            "best_pv": runs[best_idx]["best_pv"],
            "best_b_asa": runs[best_idx]["best_b_asa"],
            "best_me": runs[best_idx]["best_me"],
            "is_feasible": bool(all(r["is_feasible"] for r in runs)),
            "physics_solver_calls": calls,
            "surrogate_calls": surr,
            "wall_time_ms": round(float(np.mean(times)), 2),
            "convergence_history": runs[0]["convergence_history"],
            "per_seed_p_voo": [r["best_p_voo"] for r in runs],
        }

    std_summary = summarize_runs(std_runs)
    mf_summary = summarize_runs(mf_runs)
    eq_summary = summarize_runs(eq_runs)

    calls_saved_pct = (
        100.0
        * (std_summary["physics_solver_calls"] - mf_summary["physics_solver_calls"])
        / max(std_summary["physics_solver_calls"], 1)
    )
    speedup = std_summary["wall_time_ms"] / max(mf_summary["wall_time_ms"], 1e-3)
    score_retention_pct = 100.0 * (mf_summary["best_p_voo"] / max(std_summary["best_p_voo"], 1e-6))
    gain_over_equal_budget_pct = (
        100.0
        * (mf_summary["best_p_voo"] - eq_summary["best_p_voo"])
        / max(eq_summary["best_p_voo"], 1e-6)
    )

    return {
        "n_seeds": len(seed_list),
        "seeds": seed_list,
        "n_particles": n_particles,
        "n_iterations": n_iterations,
        "top_k_high_fidelity": top_k_high_fidelity,
        "standard_pso": std_summary,
        "multifidelity_surrogate_pso": mf_summary,
        "equal_budget_full_physics_pso": eq_summary,
        "physics_calls_reduction_pct": round(calls_saved_pct, 2),
        "wall_clock_speedup_x": round(speedup, 2),
        "score_retention_pct": round(score_retention_pct, 2),
        "gain_over_equal_budget_pct": round(gain_over_equal_budget_pct, 2),
    }
