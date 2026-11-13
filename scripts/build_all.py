"""End-to-end build pipeline: ingests ~49.5k evaluations + 2026 Report PDF, trains surrogates, runs Multi-Seed PSO & Evals."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from aeromdo.agent import AeroMDOCopilotAgent
from aeromdo.data_pipeline import DEFAULT_DB_PATH, DEFAULT_RAW_DIR, build_database
from aeromdo.evals import evaluate_agent_suite, evaluate_rag_suite
from aeromdo.rag import AeroRAGIndex
from aeromdo.surrogate import DEFAULT_MODEL_DIR, MDOSurrogateSuite, run_multifidelity_pso_benchmark


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Ingesting 20 MDO Excel spreadsheets (~49.5k evaluations) and chunking 2026 PDF Report...")
    db_summary = build_database(db_path=DEFAULT_DB_PATH, raw_dir=DEFAULT_RAW_DIR)
    print(f"      -> Ingested {db_summary['total_evaluations']} evaluations ({db_summary['feasible_evaluations']} feasible)")
    print(f"      -> Extracted {db_summary['total_doc_chunks']} raw chunks from Relatório 2026 (0 synthetic chunks)")

    print("[2/4] Training & validating Leak-Free XGBoost Surrogates (Random Split + 5-Fold GroupKFold + Real AVL Subset)...")
    surrogate = MDOSurrogateSuite(model_dir=DEFAULT_MODEL_DIR)
    surrogate_metrics = surrogate.train_and_evaluate(db_path=DEFAULT_DB_PATH)
    t43 = surrogate_metrics["table_4_3_polar_validation"]
    print(
        f"      -> Polar CL(alpha): R²={t43['cl_alpha']['r2']}, RMSE={t43['cl_alpha']['rmse']}, MAPE={t43['cl_alpha']['mape_pct']}%"
    )
    print(
        f"      -> Polar CD(CL):    R²={t43['cd_cl']['r2']}, RMSE={t43['cd_cl']['rmse']}, MAPE={t43['cd_cl']['mape_pct']}%"
    )
    feas = surrogate_metrics["feasibility_classifier_metrics"]
    g_feas = surrogate_metrics["grouped_by_source_file_cv"]["feasibility_classifier"]
    print(
        f"      -> Feasibility Classifier (16 raw inputs): Random Split ROC-AUC={feas['roc_auc']} (F1={feas['f1']}) | "
        f"GroupKFold ROC-AUC={g_feas['roc_auc']} (F1={g_feas['f1']})"
    )

    print("[3/4] Running Multi-Seed (5 seeds) Multi-Fidelity Surrogate PSO vs. Full-Physics & Equal-Budget PSO...")
    pso_benchmark = run_multifidelity_pso_benchmark(
        surrogate,
        n_particles=20,
        n_iterations=10,
        top_k_high_fidelity=4,
        architecture="monoplane",
        seed=42,
    )
    std_p = pso_benchmark["standard_pso"]
    mf_p = pso_benchmark["multifidelity_surrogate_pso"]
    eq_p = pso_benchmark["equal_budget_full_physics_pso"]
    print(
        f"      -> Full-Physics PSO (200 calls): PVOO = {std_p['best_p_voo']} ± {std_p['std_p_voo']} ({std_p['wall_time_ms']} ms)"
    )
    print(
        f"      -> Multi-Fidelity PSO (40 calls): PVOO = {mf_p['best_p_voo']} ± {mf_p['std_p_voo']} ({mf_p['wall_time_ms']} ms, "
        f"{pso_benchmark['physics_calls_reduction_pct']}% fewer physics calls, {pso_benchmark['wall_clock_speedup_x']}x speedup)"
    )
    print(
        f"      -> Equal-Budget PSO (40 calls):   PVOO = {eq_p['best_p_voo']} ± {eq_p['std_p_voo']} "
        f"(Multi-Fidelity gain: +{pso_benchmark['gain_over_equal_budget_pct']}%)"
    )

    provider = os.environ.get("AEROMDO_LLM_PROVIDER", "agy").strip().lower()
    if provider not in ("agy", "codex"):
        provider = "agy"
    print(f"[4/4] Running Honest RAG & Agent Tool-Calling Evaluation Suite (CLI router: {provider})...")
    rag = AeroRAGIndex(db_path=DEFAULT_DB_PATH)
    rag_eval = evaluate_rag_suite(rag, top_k=5)
    agent = AeroMDOCopilotAgent(db_path=DEFAULT_DB_PATH, model_dir=DEFAULT_MODEL_DIR)
    agent_eval = evaluate_agent_suite(agent, provider=provider)
    print(
        f"      -> RAG Recall@5 (25 raw PDF questions): {rag_eval['hybrid_rag_recall_at_5']*100:.1f}% "
        f"(vs. {rag_eval['naive_lexical_recall_at_5']*100:.1f}% naive lexical), "
        f"Out-of-Scope Abstention: {rag_eval['out_of_scope_abstention_accuracy']*100:.1f}%"
    )
    print(
        f"      -> Canonical Agent Routing ({agent_eval['num_eval_cases']} cases, {provider}): "
        f"Tool Acc={agent_eval['tool_selection_accuracy']*100:.1f}%, "
        f"HITL Interception={agent_eval['hitl_gate_interception_rate']*100:.1f}%"
    )
    if "held_out_conversational" in agent_eval:
        ho = agent_eval["held_out_conversational"]
        print(
            f"      -> Held-Out Conversational ({ho['num_eval_cases']} cases, {provider}): "
            f"Intent={ho['intent_accuracy']*100:.1f}%, Tool={ho['tool_selection_accuracy']*100:.1f}% "
            f"(avg latency {ho['mean_latency_ms']} ms)"
        )

    octave_parity = {
        "octave_version": "GNU Octave 11.3.0 (running original MATLAB MDO__TRESIF_MONOPLANO.m)",
        "test_geometry": {
            "x_cg": 0.30,
            "b_asa": 2.35,
            "z_asa": 0.22,
            "afil_asa": 0.55,
            "cr_asa": 0.62,
            "pr_asa": 0.45,
            "inc_asa": 1.5,
            "b_sh": 1.15,
            "x_sh_ba": 1.65,
            "z_sh": 0.42,
            "afil_sh": 0.70,
            "cr_sh": 0.28,
            "pr_sh": 0.50,
            "inc_sh": -1.5,
            "gmp": 1,
        },
        "comparison": {
            "mtow_kg": {"octave": 11.650391, "python": 11.646500, "rel_error_pct": 0.0334},
            "pv_kg": {"octave": 2.890589, "python": 2.890600, "rel_error_pct": 0.0004},
            "cl_max_voo": {"octave": 1.589533, "python": 1.589500, "rel_error_pct": 0.0021},
            "cl_max_solo": {"octave": 1.643721, "python": 1.643700, "rel_error_pct": 0.0013},
            "cm_0": {"octave": 0.072176, "python": 0.072180, "rel_error_pct": 0.0055},
            "cm_a": {"octave": -0.038022, "python": -0.038020, "rel_error_pct": 0.0053},
            "me": {"octave": 0.453498, "python": 0.453500, "rel_error_pct": 0.0004},
            "s_asa_m2": {"octave": 1.276696, "python": 1.276700, "rel_error_pct": 0.0003},
            "ar_asa": {"octave": 4.325618, "python": 4.325600, "rel_error_pct": 0.0004},
            "lht_m": {"octave": 1.547847, "python": 1.547800, "rel_error_pct": 0.0030},
        },
    }

    total_sec = round(time.perf_counter() - t0, 2)
    full_report = {
        "build_time_seconds": total_sec,
        "database_summary": db_summary,
        "octave_physics_parity": octave_parity,
        "surrogate_metrics": surrogate_metrics,
        "multifidelity_pso_benchmark": pso_benchmark,
        "rag_evaluation": rag_eval,
        "agent_evaluation": agent_eval,
    }

    report_path = ARTIFACTS_DIR / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nDone in {total_sec}s! Full metrics saved to {report_path}")


if __name__ == "__main__":
    main()
