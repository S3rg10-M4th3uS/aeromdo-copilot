"""FastAPI application and interactive Web UI for AeroMDO Copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from aeromdo.agent import AeroMDOCopilotAgent
from aeromdo.physics import run_simulation
from aeromdo.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AircraftInput,
    RAGResponse,
    SimulationResult,
    SurrogatePrediction,
)


app = FastAPI(
    title="AeroMDO Copilot API",
    version="0.1.0",
    description=(
        "Physics-Informed Surrogate Modeling, Multi-Fidelity PSO, and Agentic RAG Copilot "
        "for SAE AeroDesign Multidisciplinary Design Optimization."
    ),
)

_agent: AeroMDOCopilotAgent | None = None


def get_agent() -> AeroMDOCopilotAgent:
    global _agent
    if _agent is None:
        _agent = AeroMDOCopilotAgent()
    return _agent


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Service health and database readiness check."""
    agent = get_agent()
    db_exists = agent.db_path.exists()
    model_loaded = agent.surrogate.load() if db_exists else False
    return {
        "status": "ok",
        "database_ready": db_exists,
        "surrogate_ready": model_loaded,
    }


@app.post("/simulate", response_model=SimulationResult)
def simulate_endpoint(inp: AircraftInput) -> SimulationResult:
    """Run coupled aerodynamic, stability, empty weight (PV), and takeoff simulation."""
    return run_simulation(inp)


@app.post("/surrogate/predict", response_model=SurrogatePrediction)
def surrogate_predict_endpoint(inp: AircraftInput) -> SurrogatePrediction:
    """Run millisecond XGBoost surrogate inference with constraint feasibility probability."""
    agent = get_agent()
    return agent.surrogate.predict(inp)


@app.get("/rag/search", response_model=RAGResponse)
def rag_search_endpoint(q: str, top_k: int = 5) -> RAGResponse:
    """Query the Hybrid RAG index over the 2026 Preliminary Design Report & requirements."""
    agent = get_agent()
    return agent.rag.answer_question(q, top_k=top_k)


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat_endpoint(req: AgentChatRequest) -> AgentChatResponse:
    """Main conversational agent endpoint with tool calling, citations, and Human-in-the-Loop."""
    agent = get_agent()
    return agent.chat(req)


@app.post("/agent/approve/{ticket_id}")
def approve_ticket_endpoint(ticket_id: str) -> dict[str, Any]:
    """Approve and execute a pending Human-in-the-Loop batch optimization ticket."""
    agent = get_agent()
    try:
        return agent.approve_ticket(ticket_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index_dashboard() -> str:
    """Serve lightweight interactive web interface for AeroDesign team members."""
    return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>AeroMDO Copilot — SAE AeroDesign</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; margin-bottom: 0.2rem; }
    .sub { color: #94a3b8; margin-bottom: 1.5rem; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
    textarea, input { width: 100%; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 0.6rem; font-size: 0.95rem; }
    button { background: #0284c7; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 600; margin-top: 0.5rem; }
    button:hover { background: #0369a1; }
    pre { background: #0f172a; padding: 0.8rem; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; border: 1px solid #334155; }
    .pill { display: inline-block; background: #0369a1; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.78rem; margin-right: 0.4rem; }
  </style>
</head>
<body>
  <h1>✈️ AeroMDO Copilot</h1>
  <div class="sub">Engenharia de IA & Otimização Multidisciplinar (MDO) — 49.500 avaliações + Surrogate XGBoost + RAG Relatório 2026</div>
  <div class="card">
    <h3>Pergunte ao Agente (Banco MDO, Surrogate, Simulador Físico ou Relatório 2026)</h3>
    <textarea id="msg" rows="3">Quais configurações de monoplano com envergadura menor que 3.0 m tiveram o maior PVOO no banco de dados?</textarea>
    <div style="margin-top:0.6rem; display:flex; gap:0.6rem; align-items:center;">
      <label for="provider" style="font-size:0.9rem; color:#cbd5e1;">Roteador LLM:</label>
      <select id="provider" style="background:#0f172a; color:#f8fafc; border:1px solid #475569; border-radius:6px; padding:0.45rem;">
        <option value="agy">agy (Gemini Pro CLI - sem API key)</option>
        <option value="codex">codex (ChatGPT Pro CLI - sem API key)</option>
      </select>
      <button onclick="sendChat()" style="margin-top:0;">Enviar ao Agente</button>
    </div>
    <div id="out" style="margin-top:1rem;"></div>
  </div>
  <script>
    async function sendChat() {
      const msg = document.getElementById('msg').value;
      const provider = document.getElementById('provider').value;
      const out = document.getElementById('out');
      out.innerHTML = 'Processando...';
      const res = await fetch('/agent/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, provider: provider})
      });
      const data = await res.json();
      out.innerHTML = `<p><span class="pill">Intent: ${data.intent}</span><span class="pill">Provider: ${data.provider_used}</span><span class="pill">Latência: ${data.total_latency_ms} ms</span></p>`
        + `<p><strong>Resposta:</strong> ${data.answer}</p>`
        + `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
  </script>
</body>
</html>"""
