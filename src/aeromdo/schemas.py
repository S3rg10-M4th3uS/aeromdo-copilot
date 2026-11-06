"""Pydantic v2 schemas for aircraft design, MDO simulation, surrogate ML, RAG, and agent orchestration."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class AircraftInput(BaseModel):
    """Normalized and physical aircraft design parameters (Table 4.5, Relatório 2026 & MDO__TRESIF)."""

    architecture: Literal["monoplane", "biplane"] = Field(
        default="monoplane",
        description="Aircraft lifting surface configuration (N=1 monoplane, N=2 biplane).",
    )
    x_cg: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="CG position as fraction of Mean Aerodynamic Chord (pXCG).",
    )
    b_asa: float = Field(
        default=2.35,
        ge=1.2,
        le=4.2,
        description="Main wing span Bw [m] (REQ-SYS-001 <= 4.0 m).",
    )
    z_asa: float = Field(
        default=0.20,
        ge=0.05,
        le=0.60,
        description="Lower/main wing height above ground [m].",
    )
    z_asa2: float | None = Field(
        default=None,
        ge=0.15,
        le=0.85,
        description="Upper wing height above ground [m] (only used for biplane).",
    )
    afil_asa: float = Field(
        default=0.58,
        ge=0.35,
        le=1.0,
        description="Wing taper ratio lambda_w = Cp / Cr.",
    )
    cr_asa: float = Field(
        default=0.60,
        ge=0.20,
        le=0.90,
        description="Wing root chord CRw [m].",
    )
    pr_asa: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Wing rectangular span fraction PR_ASA.",
    )
    inc_asa: float = Field(
        default=0.0,
        ge=-5.0,
        le=6.0,
        description="Wing incidence angle Iw [deg].",
    )
    b_sh: float = Field(
        default=1.10,
        ge=0.40,
        le=2.20,
        description="Horizontal tail (EH/SH) span Bh [m].",
    )
    x_sh_ba: float = Field(
        default=1.55,
        ge=0.80,
        le=2.40,
        description="Horizontal tail leading-edge longitudinal distance from wing LE Xh [m].",
    )
    z_sh: float = Field(
        default=0.38,
        ge=0.10,
        le=0.85,
        description="Horizontal tail height above ground [m].",
    )
    afil_sh: float = Field(
        default=0.65,
        ge=0.35,
        le=1.0,
        description="Horizontal tail taper ratio.",
    )
    cr_sh: float = Field(
        default=0.24,
        ge=0.10,
        le=0.55,
        description="Horizontal tail root chord CRh [m].",
    )
    pr_sh: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Horizontal tail rectangular span fraction.",
    )
    inc_sh: float = Field(
        default=-1.0,
        ge=-6.0,
        le=4.0,
        description="Horizontal tail incidence angle Ih [deg].",
    )
    gmp: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Propulsion group archetype ID (1 = primary high-thrust setup).",
    )


class SimulationConstraints(BaseModel):
    """Inequality constraints from Table 4.4 (Relatório 2026) and MDO__FUNCAO_OBJETIVO.m."""

    ar_asa: float = Field(..., description="Wing aspect ratio AR_ASA (must be >= 4.1).")
    ar_sh: float = Field(..., description="Horizontal tail aspect ratio AR_SH (must be >= 4.1).")
    cm_0: float = Field(..., description="Zero-alpha pitching moment coefficient CM_0 (must be >= 0).")
    cm_a: float = Field(..., description="Pitch stability derivative dCm/dAlpha CM_A (must be < 0).")
    me: float = Field(..., description="Static margin ME (must be >= 0.02 in MDO / >= 0.01 in Table 4.4).")
    h_total: float = Field(..., description="Total aircraft height [m] (must be <= 0.9 m).")
    b_asa: float = Field(..., description="Wing span [m] (must be <= 4.0 m).")
    x_dec: float = Field(..., description="Takeoff ground roll distance [m] (must be <= 55.0 m).")
    penalty: float = Field(..., ge=0.0, description="Quadratic penalty value P.")
    is_feasible: bool = Field(..., description="True when all aerodynamic, stability, and geometric constraints pass.")
    violated_constraints: list[str] = Field(default_factory=list, description="List of violated constraint codes.")


class SimulationOutputs(BaseModel):
    """Aerodynamic, structural, and performance outputs from the simulator."""

    mtow: float = Field(..., description="Maximum Takeoff Weight [kg].")
    pv: float = Field(..., description="Empty Weight (Peso Vazio) [kg].")
    cp: float = Field(..., description="Payload Weight (Carga Paga = MTOW - PV) [kg].")
    ee: float = Field(..., description="Structural Efficiency EE = CP / PV.")
    p_voo: float = Field(..., description="Flight Score PVOO (Eq. 1.1: 50 * EE * f(N) * 1.15^-b).")
    objective_f: float = Field(..., description="Penalized objective function F = -P_VOO + P.")
    v_dec: float = Field(..., description="Takeoff liftoff speed V_lo [m/s].")
    x_dec: float = Field(..., description="Takeoff ground roll distance [m].")
    cl_max_solo: float = Field(..., description="Maximum lift coefficient in ground effect.")
    cl_max_voo: float = Field(..., description="Maximum lift coefficient in free flight.")
    cd_voo_cruise: float = Field(..., description="Estimated total drag coefficient CD in flight.")
    estol_av_solo: float = Field(..., description="Aircraft stall angle in ground effect [deg].")
    estol_av_voo: float = Field(..., description="Aircraft stall angle in flight [deg].")
    s_asa: float = Field(..., description="Wing reference area Sw [m^2].")
    s_sh: float = Field(..., description="Horizontal tail area Sh [m^2].")
    lht: float = Field(..., description="Tail moment arm LHT [m].")
    wing_loading: float = Field(..., description="Wing loading W/S [kg/m^2].")


class SimulationResult(BaseModel):
    """Complete output of a single physics simulation run."""

    inputs: AircraftInput
    constraints: SimulationConstraints
    outputs: SimulationOutputs
    solver: Literal["python_llt", "surrogate_xgb"] = "python_llt"
    execution_time_ms: float = Field(..., ge=0.0)


class SurrogatePrediction(BaseModel):
    """Fast ML surrogate prediction with uncertainty and constraint feasibility."""

    architecture: Literal["monoplane", "biplane"]
    solver_target: Literal["AVL", "LLT"] = "AVL"
    predicted_mtow: float = Field(..., description="Predicted MTOW [kg].")
    predicted_pv: float = Field(..., description="Predicted Empty Weight PV [kg].")
    predicted_p_voo: float = Field(..., description="Predicted Flight Score PVOO.")
    predicted_cl_max_voo: float = Field(..., description="Predicted CL_max in flight.")
    predicted_x_dec: float = Field(..., description="Predicted takeoff distance [m].")
    predicted_cm_0: float = Field(..., description="Predicted CM_0.")
    predicted_cm_a: float = Field(..., description="Predicted CM_alpha.")
    predicted_me: float = Field(..., description="Predicted Static Margin ME.")
    feasibility_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of meeting all constraints.")
    is_likely_feasible: bool = Field(..., description="True if feasibility_probability >= 0.5 and geometric bounds hold.")
    inference_time_ms: float = Field(..., ge=0.0)


class Citation(BaseModel):
    """Traceable reference to the 2026 Preliminary Design Report or SAE regulation."""

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g., 'page_23_sec_4.1.4').")
    page: int = Field(..., ge=1, description="Page number in the PDF report.")
    section: str = Field(..., description="Section title or number (e.g., '4.1.4 Modelo Aerodinâmico Surrogate (IA)').")
    requirement_ids: list[str] = Field(default_factory=list, description="Referenced requirements (e.g., ['REQ-AEROD-010']).")
    score: float = Field(..., description="Relevance score.")
    excerpt: str = Field(..., description="Direct quote excerpt from the document.")


class RAGResponse(BaseModel):
    """Structured RAG output with mandatory citations and abstention flag."""

    query: str
    abstained: bool = Field(
        ...,
        description="True when the knowledge base lacks sufficient evidence to answer reliably.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence score.")
    answer: str = Field(..., description="Grounded technical answer or explicit abstention message.")
    citations: list[Citation] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Audit record of a tool invoked by the AI Engineer Agent."""

    tool_name: Literal[
        "query_mdo_database",
        "predict_surrogate",
        "run_physics_simulation",
        "run_multifidelity_pso",
        "search_technical_docs",
    ]
    arguments: dict[str, Any]
    status: Literal["executed", "pending_human_approval", "error"]
    latency_ms: float = Field(..., ge=0.0)
    output_summary: dict[str, Any]


class HumanApprovalTicket(BaseModel):
    """Human-in-the-loop ticket created before running computationally expensive tasks."""

    ticket_id: str
    tool_name: str
    reason: str
    proposed_arguments: dict[str, Any]
    estimated_evaluations: int
    status: Literal["pending", "approved", "rejected"] = "pending"


class AgentChatRequest(BaseModel):
    """Input request to the AeroMDO Copilot Agent."""

    message: str = Field(..., min_length=2, description="User natural-language question or command.")
    provider: Literal["auto", "local_router", "gemini", "ollama"] = Field(
        default="auto",
        description="LLM provider selection ('auto' uses Gemini if GEMINI_API_KEY is set, else deterministic local_router).",
    )
    require_human_approval_for_batch: bool = Field(
        default=True,
        description="Require explicit human approval before running multi-iteration PSO or batch simulations.",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned


class AgentChatResponse(BaseModel):
    """Validated structured output returned by the AeroMDO Copilot Agent."""

    intent: Literal[
        "database_query",
        "surrogate_prediction",
        "single_simulation",
        "batch_optimization",
        "documentation_rag",
        "out_of_scope",
    ]
    answer: str = Field(..., description="Synthesized technical response in Portuguese.")
    abstained: bool = Field(default=False, description="True if query is out-of-scope or lacks evidence.")
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    approval_ticket: HumanApprovalTicket | None = None
    structured_data: dict[str, Any] = Field(default_factory=dict)
    provider_used: str = Field(default="local_router")
    total_latency_ms: float = Field(..., ge=0.0)
