"""API and agent integration checks without external data, models or CLI access."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import aeromdo.agent as agent_module
import aeromdo.api as api_module
from aeromdo.agent import AeroMDOCopilotAgent
from aeromdo.schemas import AgentChatResponse, SimulationResult


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    agent = AeroMDOCopilotAgent(
        db_path=tmp_path / "missing.db", model_dir=tmp_path / "models"
    )
    monkeypatch.setattr(api_module, "_agent", agent)
    # Fail immediately if a test accidentally invokes a real CLI.
    blocked = Mock(side_effect=AssertionError("Unexpected external CLI call"))
    monkeypatch.setattr(agent, "_route_with_agy", blocked)
    monkeypatch.setattr(agent, "_route_with_codex", blocked)
    with TestClient(api_module.app) as client:
        yield client, agent


def test_api_without_external_artifacts(isolated_client):
    client, _ = isolated_client
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "database_ready": False, "surrogate_ready": False
    }
    simulation = client.post("/simulate", json={"architecture": "monoplane"})
    assert simulation.status_code == 200
    result = SimulationResult.model_validate(simulation.json())
    assert result.outputs.mtow > result.outputs.pv > 0
    assert client.post("/simulate", json={"b_asa": -1}).status_code == 422
    assert client.post(
        "/agent/chat", json={"message": "Test", "provider": "invalid"}
    ).status_code == 422
    assert client.post("/agent/approve/unknown-ticket").status_code == 404


@pytest.mark.parametrize("provider", ["agy", "codex"])
def test_batch_waits_for_explicit_approval(isolated_client, monkeypatch, provider):
    client, agent = isolated_client
    monkeypatch.setattr(agent, f"_route_with_{provider}", lambda _: {
        "intent": "batch_optimization", "tool_name": "run_multifidelity_pso",
        "architecture": "monoplane",
    })
    benchmark = Mock(return_value={"physics_calls_reduction_pct": 80.0})
    monkeypatch.setattr(agent_module, "run_multifidelity_pso_benchmark", benchmark)
    response = client.post("/agent/chat", json={
        "message": "Rodar PSO para monoplano", "provider": provider
    })
    assert response.status_code == 200
    chat = AgentChatResponse.model_validate(response.json())
    assert chat.approval_ticket is not None
    assert chat.approval_ticket.status == "pending"
    assert chat.tool_calls[0].status == "pending_human_approval"
    benchmark.assert_not_called()

    approved = client.post(f"/agent/approve/{chat.approval_ticket.ticket_id}")
    assert approved.status_code == 200
    assert approved.json()["ticket"]["status"] == "approved"
    benchmark.assert_called_once()


def test_agent_abstains_without_running_tools(isolated_client, monkeypatch):
    client, agent = isolated_client
    monkeypatch.setattr(agent, "_route_with_agy", lambda _: {
        "intent": "out_of_scope", "tool_name": "none"
    })
    response = client.post("/agent/chat", json={"message": "Receita de bolo?"})
    assert response.status_code == 200
    chat = AgentChatResponse.model_validate(response.json())
    assert chat.abstained
    assert chat.tool_calls == []
    assert chat.citations == []

