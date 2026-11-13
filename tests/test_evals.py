"""Regression tests for Leak-Free Surrogate ML models, Honest RAG retrieval, Agent tool routing, and FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aeromdo.agent import AeroMDOCopilotAgent
from aeromdo.api import app
from aeromdo.data_pipeline import DEFAULT_DB_PATH
from aeromdo.evals import evaluate_agent_suite, evaluate_rag_suite
from aeromdo.rag import AeroRAGIndex
from aeromdo.schemas import AircraftInput
from aeromdo.surrogate import DEFAULT_MODEL_DIR, FEATURE_COLS, MDOSurrogateSuite


def test_surrogate_metrics_and_inference() -> None:
    """Verify leak-free surrogate meets Table 4.3 accuracy thresholds, includes GroupKFold & Real AVL validation, and runs in < 50 ms."""
    surrogate = MDOSurrogateSuite(model_dir=DEFAULT_MODEL_DIR)
    assert surrogate.load(), "Surrogate bundle should exist after build_all.py"

    # Verify zero derived constraint leakage in feature columns
    for leaked_col in ("ar_asa", "ar_sh", "s_asa", "s_sh", "lht"):
        assert leaked_col not in FEATURE_COLS
        assert leaked_col not in surrogate.metrics_summary["raw_feature_columns"]

    t43 = surrogate.metrics_summary["table_4_3_polar_validation"]
    assert t43["cl_alpha"]["r2"] > 0.992
    assert t43["cl_alpha"]["mape_pct"] < 1.2
    assert t43["cd_cl"]["r2"] > 0.985
    assert t43["cd_cl"]["mape_pct"] < 2.1

    # Real AVL subset validation
    avl_val = surrogate.metrics_summary["real_avl_solver_validation"]
    assert avl_val["unique_avl_rows"] >= 4500
    assert avl_val["metrics_by_target"]["cl_max_voo"]["r2"] >= 0.88
    assert avl_val["metrics_by_target"]["cm_0"]["r2"] >= 0.94

    # Deduplicated Random Split & GroupKFold by source_file
    feas = surrogate.metrics_summary["feasibility_classifier_metrics"]
    assert feas["roc_auc"] >= 0.97
    assert feas["f1"] >= 0.90

    g_cv = surrogate.metrics_summary["grouped_by_source_file_cv"]
    assert g_cv["feasibility_classifier"]["roc_auc"] >= 0.84
    assert g_cv["regression"]["cm_a"]["r2"] >= 0.85

    pred = surrogate.predict(AircraftInput())
    assert pred.predicted_mtow > 8.0
    assert pred.predicted_p_voo > 50.0
    assert 0.0 <= pred.feasibility_probability <= 1.0
    assert pred.inference_time_ms < 50.0


def test_rag_evaluation_benchmark() -> None:
    """Verify Hybrid RAG achieves high Recall@5 on raw PDF chunks, beats naive lexical search, and abstains on out-of-scope queries."""
    rag = AeroRAGIndex(db_path=DEFAULT_DB_PATH)
    metrics = evaluate_rag_suite(rag, top_k=5)
    assert metrics["num_domain_questions"] == 25
    assert metrics["hybrid_rag_recall_at_5"] >= 0.92
    assert metrics["hybrid_rag_recall_at_5"] > metrics["naive_lexical_recall_at_5"]
    assert metrics["out_of_scope_abstention_accuracy"] == 1.0


def _fake_cli_route(msg: str) -> dict[str, object]:
    """Deterministic stub simulating structured JSON output from `agy` / `codex` CLI routers."""
    lower = msg.lower()
    if any(w in lower for w in ("lasanha", "futebol", "netflix", "tesouro", "pizza")):
        return {"intent": "out_of_scope", "tool_name": "none"}
    if any(w in lower for w in ("rodar pso", "otimização com pso", "otimizacao com pso", "iniciar mdo", "enxame", "algoritmo evolucionário")):
        arch = "biplane" if "biplano" in lower else "monoplane"
        return {"intent": "batch_optimization", "tool_name": "run_multifidelity_pso", "architecture": arch}
    if any(w in lower for w in ("rodar simulação", "rodar simulacao", "simule um", "executar simulador", "calcular na simulação", "equações de llt", "física de verdade", "equações diferenciais")):
        return {"intent": "single_simulation", "tool_name": "run_physics_simulation"}
    if any(w in lower for w in ("surrogate", "predição rápida", "predicao rapida", "xgboost", "rede/modelo de ml", "metamodelo")):
        return {"intent": "surrogate_prediction", "tool_name": "predict_surrogate"}
    if any(w in lower for w in ("banco de dados", "planilhas", "histórico", "historico", "rodadas antigas", "campanhas do avl", "registros salvos")):
        return {"intent": "database_query", "tool_name": "query_mdo_database"}
    return {"intent": "documentation_rag", "tool_name": "search_technical_docs"}


def test_agent_routing_and_hitl_safety(monkeypatch) -> None:
    """Verify Agent tool routing accuracy, structured outputs, and Human-in-the-Loop gate via mocked CLI routers."""
    agent = AeroMDOCopilotAgent(db_path=DEFAULT_DB_PATH, model_dir=DEFAULT_MODEL_DIR)
    monkeypatch.setattr(agent, "_route_with_agy", _fake_cli_route)
    metrics = evaluate_agent_suite(agent, provider="agy")
    assert metrics["intent_accuracy"] >= 0.95
    assert metrics["tool_selection_accuracy"] >= 0.95
    assert metrics["structured_output_validity_rate"] == 1.0
    assert metrics["hitl_gate_interception_rate"] == 1.0
    assert metrics["held_out_conversational"]["intent_accuracy"] >= 0.95


def test_fastapi_endpoints(monkeypatch) -> None:
    """Test REST API endpoints (/health, /surrogate/predict, /rag/search, /agent/chat, /agent/approve)."""
    from aeromdo.api import get_agent

    monkeypatch.setattr(get_agent(), "_route_with_agy", _fake_cli_route)
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["database_ready"] is True

    r_pred = client.post("/surrogate/predict", json={"architecture": "monoplane", "b_asa": 2.30, "cr_asa": 0.60})
    assert r_pred.status_code == 200
    assert r_pred.json()["predicted_p_voo"] > 50.0

    r_rag = client.get("/rag/search", params={"q": "Qual o requisito REQ-ADES-002 de decolagem?"})
    assert r_rag.status_code == 200
    assert r_rag.json()["abstained"] is False

    # Test HITL ticket creation & approval via API
    r_chat = client.post(
        "/agent/chat",
        json={"message": "Rodar PSO multifidelidade para monoplano", "require_human_approval_for_batch": True},
    )
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    assert chat_data["intent"] == "batch_optimization"
    ticket_id = chat_data["approval_ticket"]["ticket_id"]

    r_approve = client.post(f"/agent/approve/{ticket_id}")
    assert r_approve.status_code == 200
    assert r_approve.json()["ticket"]["status"] == "approved"
    assert r_approve.json()["result"]["physics_calls_reduction_pct"] > 50.0


def test_cli_llm_providers_agy_and_codex(monkeypatch) -> None:
    """Verify agent routing & parameter extraction for local `agy` (Gemini Pro) and `codex` (ChatGPT Pro) CLI providers."""
    agent = AeroMDOCopilotAgent(db_path=DEFAULT_DB_PATH, model_dir=DEFAULT_MODEL_DIR)

    monkeypatch.setattr(
        agent,
        "_route_with_agy",
        lambda msg: {
            "intent": "surrogate_prediction",
            "tool_name": "predict_surrogate",
            "architecture": "monoplane",
            "b_asa": 2.30,
            "cr_asa": 0.60,
        },
    )
    monkeypatch.setattr(
        agent,
        "_route_with_codex",
        lambda msg: {
            "intent": "database_query",
            "tool_name": "query_mdo_database",
            "architecture": "biplane",
            "solver": "AVL",
            "order_by": "p_voo",
        },
    )

    from aeromdo.schemas import AgentChatRequest

    res_agy = agent.chat(AgentChatRequest(message="Estime com ML um monoplano", provider="agy"))
    assert res_agy.provider_used == "agy"
    assert res_agy.intent == "surrogate_prediction"
    assert res_agy.tool_calls[0].tool_name == "predict_surrogate"
    assert res_agy.structured_data["predicted_p_voo"] > 50.0

    res_codex = agent.chat(AgentChatRequest(message="Mostre os biplanos no AVL", provider="codex"))
    assert res_codex.provider_used == "codex"
    assert res_codex.intent == "database_query"
    assert res_codex.tool_calls[0].tool_name == "query_mdo_database"
    assert res_codex.structured_data["matching_count"] > 0

    import pytest
    from pydantic import ValidationError

    for removed_provider in ("local_router", "gemini", "auto", "ollama"):
        with pytest.raises(ValidationError):
            AgentChatRequest(message="Teste", provider=removed_provider)  # type: ignore[arg-type]

    monkeypatch.setattr(agent, "_route_with_agy", lambda msg: None)
    with pytest.raises(RuntimeError):
        agent.chat(AgentChatRequest(message="Teste falha CLI", provider="agy"))
