"""Agentic Orchestrator with Tool Calling, Human-in-the-Loop (HITL), and Structured Outputs.

Supports exclusively local subscription CLI routers:
- Antigravity CLI (`agy --print`, uses Gemini Pro subscription locally without API key)
- Codex CLI (`codex exec`, uses ChatGPT Pro subscription locally without API key)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from aeromdo.data_pipeline import DEFAULT_DB_PATH
from aeromdo.physics import run_simulation
from aeromdo.rag import AeroRAGIndex
from aeromdo.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AircraftInput,
    Citation,
    HumanApprovalTicket,
    ToolCallRecord,
)
from aeromdo.surrogate import DEFAULT_MODEL_DIR, MDOSurrogateSuite, run_multifidelity_pso_benchmark


class AeroMDOCopilotAgent:
    """AI Engineer Copilot orchestrating MDO database queries, surrogate ML, physics simulation, and RAG."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        model_dir: Path = DEFAULT_MODEL_DIR,
    ) -> None:
        self.db_path = db_path
        self.surrogate = MDOSurrogateSuite(model_dir=model_dir)
        self.rag = AeroRAGIndex(db_path=db_path)
        self.pending_tickets: dict[str, HumanApprovalTicket] = {}

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------
    def tool_query_mdo_database(
        self,
        architecture: str | None = None,
        solver: str | None = None,
        optimizer: str | None = None,
        max_b_asa: float | None = None,
        min_me: float | None = None,
        feasible_only: bool = True,
        order_by: str = "p_voo",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Safe parameterized query over the ~51k historical MDO evaluations."""
        clauses: list[str] = []
        params: list[Any] = []

        if feasible_only:
            clauses.append("is_feasible = 1")
        if architecture in ("monoplane", "biplane"):
            clauses.append("architecture = ?")
            params.append(architecture)
        if solver in ("AVL", "LLT"):
            clauses.append("solver = ?")
            params.append(solver)
        if optimizer in ("PSO", "PatternSearch"):
            clauses.append("optimizer = ?")
            params.append(optimizer)
        if max_b_asa is not None:
            clauses.append("b_asa <= ?")
            params.append(float(max_b_asa))
        if min_me is not None:
            clauses.append("me >= ?")
            params.append(float(min_me))

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sort_col = "mtow" if order_by.lower() == "mtow" else "p_voo"
        safe_limit = max(1, min(int(limit), 20))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            count_row = cur.execute(
                f"SELECT COUNT(*) AS n, AVG(mtow) AS avg_mtow, MAX(mtow) AS max_mtow, "
                f"AVG(p_voo) AS avg_pvoo, MAX(p_voo) AS max_pvoo FROM mdo_evaluations {where_sql}",
                params,
            ).fetchone()

            rows = cur.execute(
                f"SELECT source_file, architecture, solver, optimizer, mtow, pv, cp, ee, p_voo, "
                f"b_asa, cr_asa, afil_asa, b_sh, cr_sh, x_sh_ba, ar_asa, me, cm_0, cm_a, x_dec "
                f"FROM mdo_evaluations {where_sql} ORDER BY {sort_col} DESC LIMIT ?",
                [*params, safe_limit],
            ).fetchall()

        top_designs = [dict(r) for r in rows]
        return {
            "matching_count": int(count_row["n"] or 0),
            "avg_mtow": round(float(count_row["avg_mtow"] or 0.0), 3),
            "max_mtow": round(float(count_row["max_mtow"] or 0.0), 3),
            "avg_p_voo": round(float(count_row["avg_pvoo"] or 0.0), 3),
            "max_p_voo": round(float(count_row["max_pvoo"] or 0.0), 3),
            "filters_applied": {
                "architecture": architecture,
                "solver": solver,
                "optimizer": optimizer,
                "max_b_asa": max_b_asa,
                "min_me": min_me,
                "feasible_only": feasible_only,
                "order_by": sort_col,
            },
            "top_designs": top_designs,
        }

    def tool_predict_surrogate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run millisecond XGBoost surrogate prediction for candidate geometry."""
        inp = AircraftInput(**params)
        pred = self.surrogate.predict(inp)
        return pred.model_dump()

    def tool_run_physics_simulation(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run coupled aerodynamic, stability, empty weight, and takeoff simulation."""
        inp = AircraftInput(**params)
        res = run_simulation(inp)
        return res.model_dump()

    def tool_search_technical_docs(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search 2026 Report and SAE AeroDesign requirements using Hybrid RAG."""
        rag_res = self.rag.answer_question(query, top_k=top_k)
        return rag_res.model_dump()

    # -------------------------------------------------------------------------
    # Parameter & Intent Extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_aircraft_params_from_text(text: str) -> dict[str, Any]:
        """Extract geometric parameters mentioned in natural language (Portuguese or English)."""
        lower = text.lower()
        params: dict[str, Any] = {}

        if "biplano" in lower or "biplane" in lower:
            params["architecture"] = "biplane"
        elif "monoplano" in lower or "monoplane" in lower:
            params["architecture"] = "monoplane"

        patterns = [
            (r"(?:b_asa|bw|envergadura(?:\s+da\s+asa)?)\s*(?:de|=|:)?\s*(\d+[.,]?\d*)\s*m?", "b_asa"),
            (r"(?:cr_asa|crw|corda\s+(?:na\s+)?raiz(?:\s+da\s+asa)?)\s*(?:de|=|:)?\s*(\d+[.,]?\d*)\s*m?", "cr_asa"),
            (r"(?:afil_asa|afilamento(?:\s+da\s+asa)?)\s*(?:de|=|:)?\s*(\d+[.,]?\d*)", "afil_asa"),
            (r"(?:b_sh|bh|envergadura\s+da\s+(?:eh|sh|empenagem))\s*(?:de|=|:)?\s*(\d+[.,]?\d*)\s*m?", "b_sh"),
            (r"(?:cr_sh|crh|corda\s+da\s+(?:eh|sh|empenagem))\s*(?:de|=|:)?\s*(\d+[.,]?\d*)\s*m?", "cr_sh"),
            (r"(?:x_sh_ba|xh|dist[aâ]ncia\s+da\s+(?:eh|sh|cauda))\s*(?:de|=|:)?\s*(\d+[.,]?\d*)\s*m?", "x_sh_ba"),
            (r"(?:x_cg|pxcg|cg)\s*(?:de|=|:)?\s*(\d+[.,]?\d*)", "x_cg"),
            (r"(?:z_asa|altura\s+da\s+asa)\s*(?:de|=|:)?\s*(\d+[.,]?\d*)", "z_asa"),
            (r"(?:z_sh|altura\s+da\s+(?:eh|sh))\s*(?:de|=|:)?\s*(\d+[.,]?\d*)", "z_sh"),
        ]
        for pat, key in patterns:
            m = re.search(pat, lower)
            if m:
                val = float(m.group(1).replace(",", "."))
                # Convert cm to m if user wrote e.g. 60 cm
                if key in ("cr_asa", "cr_sh") and val > 2.0:
                    val = val / 100.0
                if key == "x_cg" and val > 1.0:
                    val = val / 100.0
                params[key] = val

        return params

    ROUTER_SYSTEM_INSTRUCTION = (
        "Você é o roteador de ferramentas (Tool Router) do AeroMDO Copilot, um sistema de engenharia aeronáutica "
        "para a competição SAE BRASIL AeroDesign 2026.\n"
        "Classifique a solicitação do usuário em exatamente uma das 6 intenções e escolha a ferramenta correspondente:\n"
        "1. intent='database_query', tool_name='query_mdo_database': quando o usuário quiser consultar o banco de dados "
        "histórico (~49,5 mil avaliações salvas nas planilhas Excel), filtrar os melhores indivíduos já avaliados, "
        "comparar campanhas passadas (AVL vs LLT, PSO vs PatternSearch, monoplano vs biplano) ou ver o maior MTOW/PVOO salvo.\n"
        "2. intent='surrogate_prediction', tool_name='predict_surrogate': quando o usuário pedir uma predição/estimativa "
        "rápida usando o metamodelo de Machine Learning (surrogate / XGBoost) para uma geometria específica.\n"
        "3. intent='single_simulation', tool_name='run_physics_simulation': quando o usuário pedir para simular/calcular "
        "uma geometria específica diretamente no simulador físico acoplado (LLT, estabilidade, peso vazio, corrida de decolagem).\n"
        "4. intent='batch_optimization', tool_name='run_multifidelity_pso': quando o usuário pedir para rodar/executar uma "
        "nova busca/otimização por enxame de partículas (PSO / MDO multifidelidade) para encontrar uma aeronave ótima.\n"
        "5. intent='documentation_rag', tool_name='search_technical_docs': quando o usuário fizer perguntas conceituais, "
        "de requisitos (REQ-SYS, REQ-AEROD, REQ-ESTC, REQ-ADES, DG), equações, tabelas, metodologia ou justificativas "
        "do Relatório de Projeto 2026 ou do regulamento SAE AeroDesign.\n"
        "6. intent='out_of_scope', tool_name='none': quando a pergunta não tiver relação com engenharia aeronáutica, "
        "MDO, física de voo ou SAE AeroDesign (ex: culinária, futebol, investimentos financeiros, cinema, etc.).\n"
        "Extraia também parâmetros numéricos se mencionados (b_asa em metros, cr_asa em metros, afil_asa, b_sh, cr_sh, x_sh_ba, x_cg; "
        "use null quando um campo numérico não for informado)."
    )

    STANDARD_ROUTER_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "documentation_rag",
                    "database_query",
                    "surrogate_prediction",
                    "single_simulation",
                    "batch_optimization",
                    "out_of_scope",
                ],
            },
            "tool_name": {
                "type": "string",
                "enum": [
                    "search_technical_docs",
                    "query_mdo_database",
                    "predict_surrogate",
                    "run_physics_simulation",
                    "run_multifidelity_pso",
                    "none",
                ],
            },
            "architecture": {"type": "string", "enum": ["monoplane", "biplane", "unspecified"]},
            "solver": {"type": "string", "enum": ["AVL", "LLT", "unspecified"]},
            "optimizer": {"type": "string", "enum": ["PSO", "PatternSearch", "unspecified"]},
            "b_asa": {"type": ["number", "null"]},
            "cr_asa": {"type": ["number", "null"]},
            "afil_asa": {"type": ["number", "null"]},
            "b_sh": {"type": ["number", "null"]},
            "cr_sh": {"type": ["number", "null"]},
            "x_sh_ba": {"type": ["number", "null"]},
            "x_cg": {"type": ["number", "null"]},
            "max_b_asa": {"type": ["number", "null"]},
            "order_by": {"type": "string", "enum": ["p_voo", "mtow"]},
        },
        "required": [
            "intent",
            "tool_name",
            "architecture",
            "solver",
            "optimizer",
            "b_asa",
            "cr_asa",
            "afil_asa",
            "b_sh",
            "cr_sh",
            "x_sh_ba",
            "x_cg",
            "max_b_asa",
            "order_by",
        ],
    }

    @classmethod
    def _route_with_agy(cls, message: str) -> dict[str, Any] | None:
        """Route intent & extract parameters via installed `agy` CLI (Gemini Pro subscription, no API key)."""
        agy_bin = shutil.which("agy")
        if not agy_bin:
            return None
        prompt = (
            f"{cls.ROUTER_SYSTEM_INSTRUCTION}\n\n"
            f"Solicitação do usuário: {message}\n"
            "Responda exclusivamente com o objeto JSON estruturado seguindo o schema."
        )
        try:
            proc = subprocess.run(
                [
                    agy_bin,
                    "--print",
                    prompt,
                    "--output-format",
                    "json",
                    "--json-schema",
                    json.dumps(cls.STANDARD_ROUTER_SCHEMA, ensure_ascii=False),
                    "--effort",
                    "low",
                    "--disable-slash-commands",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            payload = json.loads(proc.stdout.strip())
            if isinstance(payload.get("structured_output"), dict):
                return payload["structured_output"]
            if isinstance(payload.get("response"), str):
                return json.loads(payload["response"].strip())
        except Exception:
            return None
        return None

    @classmethod
    def _route_with_codex(cls, message: str) -> dict[str, Any] | None:
        """Route intent & extract parameters via installed `codex exec` CLI (ChatGPT Pro subscription, no API key)."""
        codex_bin = shutil.which("codex")
        if not codex_bin:
            return None
        prompt = (
            f"{cls.ROUTER_SYSTEM_INSTRUCTION}\n\n"
            f"Solicitação do usuário: {message}\n"
            "Não execute comandos no terminal; apenas retorne o JSON final conforme o output-schema."
        )
        schema_path: Path | None = None
        out_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as sf:
                json.dump(cls.STANDARD_ROUTER_SCHEMA, sf, ensure_ascii=False)
                schema_path = Path(sf.name)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as of:
                out_path = Path(of.name)

            proc = subprocess.run(
                [
                    codex_bin,
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "-s",
                    "read-only",
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(out_path),
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if proc.returncode == 0 and out_path.exists():
                raw = out_path.read_text(encoding="utf-8").strip()
                if raw:
                    return json.loads(raw)
        except Exception:
            return None
        finally:
            if schema_path and schema_path.exists():
                schema_path.unlink(missing_ok=True)
            if out_path and out_path.exists():
                out_path.unlink(missing_ok=True)
        return None

    # -------------------------------------------------------------------------
    # Main Orchestration Entrypoint
    # -------------------------------------------------------------------------
    def chat(self, req: AgentChatRequest) -> AgentChatResponse:
        """Process a user message, invoke appropriate tools, enforce HITL, and return structured output."""
        t0 = time.perf_counter()
        if req.provider == "agy":
            llm_route = self._route_with_agy(req.message)
        elif req.provider == "codex":
            llm_route = self._route_with_codex(req.message)
        else:
            raise ValueError(f"Unsupported router provider '{req.provider}'. Expected 'agy' or 'codex'.")

        if not llm_route or "intent" not in llm_route:
            raise RuntimeError(
                f"LLM router '{req.provider}' failed to return a valid structured intent."
            )

        intent = str(llm_route["intent"])
        provider_used = req.provider

        tool_calls: list[ToolCallRecord] = []
        citations: list[Citation] = []
        structured_data: dict[str, Any] = {}
        approval_ticket: HumanApprovalTicket | None = None

        if intent == "out_of_scope":
            elapsed = (time.perf_counter() - t0) * 1000.0
            return AgentChatResponse(
                intent="out_of_scope",
                answer=(
                    "Abstenção: Esta solicitação está fora do escopo técnico do AeroMDO Copilot "
                    "(projeto aeronáutico SAE AeroDesign, simulação MDO, metamodelos surrogate e Relatório 2026)."
                ),
                abstained=True,
                tool_calls=[],
                citations=[],
                structured_data={},
                provider_used=provider_used,
                total_latency_ms=round(elapsed, 3),
            )

        if intent == "batch_optimization":
            params = self.extract_aircraft_params_from_text(req.message)
            if llm_route and llm_route.get("architecture") in ("monoplane", "biplane"):
                params["architecture"] = llm_route["architecture"]
            arch = params.get("architecture", "monoplane")
            proposed_args = {
                "architecture": arch,
                "n_particles": 20,
                "n_iterations": 10,
                "top_k_high_fidelity": 4,
            }
            if req.require_human_approval_for_batch:
                ticket_id = f"hitl-{uuid.uuid4().hex[:8]}"
                approval_ticket = HumanApprovalTicket(
                    ticket_id=ticket_id,
                    tool_name="run_multifidelity_pso",
                    reason=(
                        "Otimização PSO executa múltiplas simulações físicas acopladas. "
                        "Requer aprovação humana antes de iniciar o job em lote."
                    ),
                    proposed_arguments=proposed_args,
                    estimated_evaluations=10 * 4,
                    status="pending",
                )
                self.pending_tickets[ticket_id] = approval_ticket
                tool_calls.append(
                    ToolCallRecord(
                        tool_name="run_multifidelity_pso",
                        arguments=proposed_args,
                        status="pending_human_approval",
                        latency_ms=0.5,
                        output_summary={"ticket_id": ticket_id, "status": "pending_human_approval"},
                    )
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                return AgentChatResponse(
                    intent="batch_optimization",
                    answer=(
                        f"Ação pausada para aprovação humana (Human-in-the-Loop): O job `run_multifidelity_pso` "
                        f"para arquitetura `{arch}` (20 partículas, 10 iterações, top-4 alta fidelidade = 40 chamadas "
                        f"ao simulador físico) aguarda confirmação no ticket `{ticket_id}`."
                    ),
                    abstained=False,
                    tool_calls=tool_calls,
                    approval_ticket=approval_ticket,
                    structured_data={"ticket": approval_ticket.model_dump()},
                    provider_used=provider_used,
                    total_latency_ms=round(elapsed, 3),
                )

            t_tool = time.perf_counter()
            pso_res = run_multifidelity_pso_benchmark(
                self.surrogate,
                n_particles=16,
                n_iterations=8,
                top_k_high_fidelity=4,
                architecture=arch,
            )
            t_lat = (time.perf_counter() - t_tool) * 1000.0
            tool_calls.append(
                ToolCallRecord(
                    tool_name="run_multifidelity_pso",
                    arguments=proposed_args,
                    status="executed",
                    latency_ms=round(t_lat, 3),
                    output_summary={
                        "best_p_voo": pso_res["multifidelity_surrogate_pso"]["best_p_voo"],
                        "physics_calls_reduction_pct": pso_res["physics_calls_reduction_pct"],
                    },
                )
            )
            structured_data = pso_res
            mf = pso_res["multifidelity_surrogate_pso"]
            answer = (
                f"Otimização PSO Multifidelidade concluída: Melhor PVOO = {mf['best_p_voo']:.2f} "
                f"(MTOW = {mf['best_mtow']:.2f} kg, PV = {mf['best_pv']:.2f} kg, Bw = {mf['best_b_asa']:.2f} m), "
                f"reduzindo em {pso_res['physics_calls_reduction_pct']:.1f}% as chamadas ao simulador físico "
                f"(speedup de {pso_res['wall_clock_speedup_x']:.2f}x)."
            )

        elif intent == "single_simulation":
            params = self.extract_aircraft_params_from_text(req.message)
            if llm_route:
                if llm_route.get("architecture") in ("monoplane", "biplane"):
                    params.setdefault("architecture", llm_route["architecture"])
                for k in ("b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba", "x_cg"):
                    if isinstance(llm_route.get(k), (int, float)) and llm_route[k] > 0:
                        params.setdefault(k, float(llm_route[k]))
            t_tool = time.perf_counter()
            sim_out = self.tool_run_physics_simulation(params)
            t_lat = (time.perf_counter() - t_tool) * 1000.0
            tool_calls.append(
                ToolCallRecord(
                    tool_name="run_physics_simulation",
                    arguments=params,
                    status="executed",
                    latency_ms=round(t_lat, 3),
                    output_summary={
                        "mtow": sim_out["outputs"]["mtow"],
                        "pv": sim_out["outputs"]["pv"],
                        "p_voo": sim_out["outputs"]["p_voo"],
                        "is_feasible": sim_out["constraints"]["is_feasible"],
                    },
                )
            )
            structured_data = sim_out
            out = sim_out["outputs"]
            con = sim_out["constraints"]
            viols = ", ".join(con["violated_constraints"]) if con["violated_constraints"] else "Nenhuma (100% viável)"
            answer = (
                f"Simulação física acoplada concluída ({sim_out['execution_time_ms']:.1f} ms): "
                f"MTOW = {out['mtow']:.2f} kg, Peso Vazio (PV) = {out['pv']:.2f} kg, "
                f"Carga Paga (CP) = {out['cp']:.2f} kg, Eficiência Estrutural (EE) = {out['ee']:.2f}, "
                f"Pontuação de Voo (PVOO) = {out['p_voo']:.2f}, Corrida de Decolagem = {out['x_dec']:.1f} m, "
                f"Margem Estática (ME) = {con['me'] * 100:.2f}%. Restrições violadas: {viols}."
            )

        elif intent == "surrogate_prediction":
            params = self.extract_aircraft_params_from_text(req.message)
            if llm_route:
                if llm_route.get("architecture") in ("monoplane", "biplane"):
                    params.setdefault("architecture", llm_route["architecture"])
                for k in ("b_asa", "cr_asa", "afil_asa", "b_sh", "cr_sh", "x_sh_ba", "x_cg"):
                    if isinstance(llm_route.get(k), (int, float)) and llm_route[k] > 0:
                        params.setdefault(k, float(llm_route[k]))
            t_tool = time.perf_counter()
            pred_out = self.tool_predict_surrogate(params)
            t_lat = (time.perf_counter() - t_tool) * 1000.0
            tool_calls.append(
                ToolCallRecord(
                    tool_name="predict_surrogate",
                    arguments=params,
                    status="executed",
                    latency_ms=round(t_lat, 3),
                    output_summary={
                        "predicted_p_voo": pred_out["predicted_p_voo"],
                        "predicted_mtow": pred_out["predicted_mtow"],
                        "feasibility_probability": pred_out["feasibility_probability"],
                    },
                )
            )
            structured_data = pred_out
            answer = (
                f"Predição do Metamodelo Surrogate XGBoost ({pred_out['inference_time_ms']:.2f} ms): "
                f"PVOO previsto = {pred_out['predicted_p_voo']:.2f}, MTOW = {pred_out['predicted_mtow']:.2f} kg, "
                f"PV = {pred_out['predicted_pv']:.2f} kg, CLmax (voo) = {pred_out['predicted_cl_max_voo']:.3f}, "
                f"Decolagem = {pred_out['predicted_x_dec']:.1f} m, ME = {pred_out['predicted_me'] * 100:.2f}%, "
                f"Probabilidade de Viabilidade = {pred_out['feasibility_probability'] * 100:.1f}%."
            )

        elif intent == "database_query":
            lower = req.message.lower()
            arch = "monoplane" if "monoplano" in lower else ("biplane" if "biplano" in lower else None)
            solver = "AVL" if "avl" in lower and "llt" not in lower else ("LLT" if "llt" in lower and "avl" not in lower else None)
            opt = "PSO" if "pso" in lower else ("PatternSearch" if "pattern" in lower else None)
            m_b = re.search(r"envergadura\s*(?:menor\s+que|<=|<|até)\s*(\d+[.,]?\d*)", lower)
            max_b = float(m_b.group(1).replace(",", ".")) if m_b else None
            order_by = "mtow" if "maior mtow" in lower else "p_voo"

            if llm_route:
                if arch is None and llm_route.get("architecture") in ("monoplane", "biplane"):
                    arch = llm_route["architecture"]
                if solver is None and llm_route.get("solver") in ("AVL", "LLT"):
                    solver = llm_route["solver"]
                if opt is None and llm_route.get("optimizer") in ("PSO", "PatternSearch"):
                    opt = llm_route["optimizer"]
                if max_b is None and isinstance(llm_route.get("max_b_asa"), (int, float)) and llm_route["max_b_asa"] > 0:
                    max_b = float(llm_route["max_b_asa"])
                if llm_route.get("order_by") in ("mtow", "p_voo"):
                    order_by = llm_route["order_by"]

            q_args = {
                "architecture": arch,
                "solver": solver,
                "optimizer": opt,
                "max_b_asa": max_b,
                "feasible_only": "inviáv" not in lower,
                "order_by": order_by,
                "limit": 5,
            }
            t_tool = time.perf_counter()
            db_out = self.tool_query_mdo_database(**q_args)
            t_lat = (time.perf_counter() - t_tool) * 1000.0
            tool_calls.append(
                ToolCallRecord(
                    tool_name="query_mdo_database",
                    arguments=q_args,
                    status="executed",
                    latency_ms=round(t_lat, 3),
                    output_summary={
                        "matching_count": db_out["matching_count"],
                        "max_p_voo": db_out["max_p_voo"],
                        "max_mtow": db_out["max_mtow"],
                    },
                )
            )
            structured_data = db_out
            if db_out["top_designs"]:
                best = db_out["top_designs"][0]
                answer = (
                    f"Consulta ao banco MDO ({db_out['matching_count']} avaliações encontradas): "
                    f"Melhor indivíduo em `{best['source_file']}` ({best['architecture']}, solver {best['solver']}) "
                    f"atingiu PVOO = {best['p_voo']:.2f} e MTOW = {best['mtow']:.2f} kg "
                    f"(PV = {best['pv']:.2f} kg, Bw = {best['b_asa']:.2f} m, CRw = {best['cr_asa']:.2f} m, "
                    f"ME = {best['me'] * 100:.2f}%, X_dec = {best['x_dec']:.1f} m)."
                )
            else:
                answer = "Nenhum indivíduo encontrado no banco de dados com os filtros especificados."

        else:
            # documentation_rag
            t_tool = time.perf_counter()
            rag_out = self.tool_search_technical_docs(req.message, top_k=5)
            t_lat = (time.perf_counter() - t_tool) * 1000.0
            tool_calls.append(
                ToolCallRecord(
                    tool_name="search_technical_docs",
                    arguments={"query": req.message, "top_k": 5},
                    status="executed",
                    latency_ms=round(t_lat, 3),
                    output_summary={
                        "abstained": rag_out["abstained"],
                        "confidence": rag_out["confidence"],
                        "top_chunk": rag_out["citations"][0]["chunk_id"] if rag_out["citations"] else None,
                    },
                )
            )
            structured_data = rag_out
            citations = [Citation(**c) for c in rag_out["citations"]]
            answer = rag_out["answer"]

            if rag_out["abstained"]:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return AgentChatResponse(
                    intent="documentation_rag",
                    answer=answer,
                    abstained=True,
                    tool_calls=tool_calls,
                    citations=[],
                    structured_data=structured_data,
                    provider_used=provider_used,
                    total_latency_ms=round(elapsed, 3),
                )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return AgentChatResponse(
            intent=intent,  # type: ignore[arg-type]
            answer=answer,
            abstained=False,
            tool_calls=tool_calls,
            citations=citations,
            approval_ticket=approval_ticket,
            structured_data=structured_data,
            provider_used=provider_used,
            total_latency_ms=round(elapsed, 3),
        )

    def approve_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Approve and execute a pending Human-in-the-Loop optimization ticket."""
        if ticket_id not in self.pending_tickets:
            raise KeyError(f"Ticket {ticket_id} not found.")
        ticket = self.pending_tickets[ticket_id]
        ticket.status = "approved"
        args = ticket.proposed_arguments
        res = run_multifidelity_pso_benchmark(
            self.surrogate,
            n_particles=min(int(args.get("n_particles", 12)), 12),
            n_iterations=min(int(args.get("n_iterations", 5)), 5),
            top_k_high_fidelity=min(int(args.get("top_k_high_fidelity", 3)), 3),
            architecture=str(args.get("architecture", "monoplane")),
            seeds=[42],
        )
        return {"ticket": ticket.model_dump(), "result": res}
