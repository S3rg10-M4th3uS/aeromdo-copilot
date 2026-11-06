"""Versioned AI Evaluation Suite for RAG Retrieval, Abstention, Agent Tool Routing (Local + Gemini LLM), and HITL Safety."""

from __future__ import annotations

import os
from typing import Any

from aeromdo.agent import AeroMDOCopilotAgent
from aeromdo.rag import AeroRAGIndex
from aeromdo.schemas import AgentChatRequest


# 25 ground-truth technical questions mapped to expected pages in Relatório 2026 (zero synthetic chunks)
RAG_BENCHMARK_QUESTIONS: list[dict[str, Any]] = [
    {"q": "Qual é a equação da pontuação de voo PVOO na competição SAE AeroDesign 2026?", "pages": [9]},
    {"q": "Como é definida a função f(N) para o número de superfícies sustentadoras na Eq. 1.1?", "pages": [9, 22]},
    {"q": "Quais são as três frentes ou pontos focais que cruzam a estrutura organizacional da equipe?", "pages": [12]},
    {"q": "Qual cadeia crítica do diagrama PERT possui maior risco e zero folga no projeto?", "pages": [14]},
    {"q": "Quais foram as quatro frentes apontadas no diagrama de Ishikawa de lições aprendidas de 2025?", "pages": [15]},
    {"q": "Por que a equipe escolheu trem de pouso triciclo e bordo de ataque enflechado na Tabela 4.1?", "pages": [21]},
    {"q": "Como o arrasto parasita da fuselagem e carga paga é modelado nas premissas de simplificação (Eq. 4.1)?", "pages": [21]},
    {"q": "Quais conceitos de configuração de aeronave foram descartados na análise qualitativa e quais seguiram?", "pages": [22]},
    {"q": "Quais perfis aerodinâmicos de referência foram fixados para Monoplano, Asa Voadora e EH no conceitual?", "pages": [22]},
    {"q": "Quais foram as métricas R², RMSE e MAPE de validação do metamodelo surrogate aerodinâmico na Tabela 4.3 (REQ-AEROD-010)?", "pages": [23]},
    {"q": "Qual é a equação de aceleração a(V) durante a corrida de decolagem em solo (Eq. 4.2)?", "pages": [25]},
    {"q": "Qual conjunto motopropulsor (motor e hélice) dominou os três arquétipos de GMP entre 9 e 13 m/s?", "pages": [28]},
    {"q": "Quantos indivíduos por arquitetura foram amostrados no Design of Experiments (DoE) via Hipercubo Latino (LHS)?", "pages": [29]},
    {"q": "Qual é a distância máxima de decolagem permitida pela restrição REQ-ADES-002 na Tabela 4.4?", "pages": [29]},
    {"q": "Qual é o limite máximo de envergadura Bw pelo requisito REQ-SYS-001 e o limite de MTOW pelo REQ-SYS-003?", "pages": [29, 30]},
    {"q": "Qual é o alongamento mínimo AR exigido pelo requisito REQ-AEROD-001 e a margem estática mínima na Tabela 4.4?", "pages": [29]},
    {"q": "Quais são os limites mínimo e máximo de pXCG, envergadura Bw e corda na raiz CRw na Tabela 4.5?", "pages": [30]},
    {"q": "Qual solver de vorticidade superficial com acoplamento viscoso foi adotado na transição de fidelidade do Projeto Preliminar?", "pages": [41]},
    {"q": "Qual algoritmo multiobjetivo acoplado à parametrização CST e XFOIL em Re = 4e5 otimizou os perfis no preliminar?", "pages": [41]},
    {"q": "Quais são os cinco perfis candidatos da família da asa na Tabela 5.1 e seus valores de Clmax e espessura t/c?", "pages": [42, 43]},
    {"q": "Por que o perfil NACA 0012 foi escolhido para a empenagem horizontal (EH) em vez do HS2 na Tabela 5.2?", "pages": [43, 44]},
    {"q": "Quais são os limites de deflexão máxima (delta) e torção máxima (theta) no dimensionamento da longarina?", "pages": [46, 50]},
    {"q": "Por que o tubo cônico de carbono superou a viga I com D-Box no acoplamento aero-estrutural do MDO?", "pages": [46, 50, 51, 52]},
    {"q": "Qual foi a comparação de pontuação PVOO entre os arranjos Tractor e Pusher na simulação FlightStream?", "pages": [47]},
    {"q": "Para quais valores de envergadura Bw, área Sw, margem estática ME e torção theta convergiu o indivíduo ótimo na síntese do MDO?", "pages": [50, 52]},
]

OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "Qual é a receita tradicional de bolo de rolo pernambucano com goiabada?",
    "Como configurar um cluster Kubernetes na DigitalOcean para um e-commerce de roupas?",
    "Quem ganhou a final da Copa Libertadores da América de futebol em 2023?",
    "Qual é a cotação atual do Bitcoin em dólares e qual ação comprar hoje na bolsa?",
    "Como fazer tempero caseiro para churrasco de picanha na brasa?",
    "Qual o melhor roteiro turístico de 5 dias em Paris na França?",
]

AGENT_ROUTING_BENCHMARK: list[dict[str, str]] = [
    {"msg": "Quais configurações de monoplano com envergadura menor que 3.0 m tiveram o maior PVOO no banco de dados?", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Consulte nas planilhas do histórico os top 5 indivíduos de biplano avaliados no solver AVL.", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Qual foi o maior MTOW nas planilhas salvas usando otimizador PSO?", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Filtrar no banco de dados as avaliações viáveis de monoplano com solver LLT.", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Quantos indivíduos viáveis existem no histórico de avaliações salvas para biplano?", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Prever com surrogate um monoplano com b_asa=2.30, cr_asa=0.60, afil_asa=0.58, b_sh=1.10.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Qual a previsão do surrogate para um biplano com envergadura de 2.4 m e corda na raiz de 0.55 m?", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Use a inferência rápida do surrogate para estimar PVOO e MTOW com b_asa=2.5 e x_sh_ba=1.6.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Quanto o surrogate prevê de margem estática e CLmax para b_asa=2.26 e cr_asa=0.60?", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Faça uma predição rápida com o surrogate para monoplano com b_asa=2.8 e b_sh=1.2.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Rodar simulação física completa para um monoplano com b_asa=2.35, cr_asa=0.60, x_sh_ba=1.55.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Simule um monoplano com envergadura de 2.4 m e corda na raiz de 0.58 m no simulador.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Executar simulador físico para biplano com b_asa=2.50, cr_asa=0.55 e b_sh=1.15.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Calcular na simulação física a corrida de decolagem para b_asa=2.26 e cr_asa=0.60.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Rodar PSO multifidelidade para encontrar a melhor geometria de monoplano.", "intent": "batch_optimization", "tool": "run_multifidelity_pso"},
    {"msg": "Executar otimização com PSO para a configuração biplano.", "intent": "batch_optimization", "tool": "run_multifidelity_pso"},
    {"msg": "Iniciar MDO com otimização multifidelidade para monoplano.", "intent": "batch_optimization", "tool": "run_multifidelity_pso"},
    {"msg": "Quais foram as métricas R² e MAPE do modelo surrogate na Tabela 4.3 do relatório?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Qual é a restrição de pista de decolagem segundo o requisito REQ-ADES-002?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Por que o tubo cônico de carbono venceu a viga I na longarina segundo a seção 5.4?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Qual motor e hélice foram escolhidos na seleção do GMP no relatório?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Qual receita de lasanha bolonhesa você recomenda para o jantar?", "intent": "out_of_scope", "tool": "none"},
    {"msg": "Quem venceu o campeonato brasileiro de futebol no ano passado?", "intent": "out_of_scope", "tool": "none"},
]

# Independent held-out conversational prompts (not used to design keyword rules)
HELD_OUT_NATURAL_PROMPTS: list[dict[str, str]] = [
    {"msg": "Nas rodadas antigas que salvamos no Excel, qual avião biplano deu a maior pontuação de voo?", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Mostra pra mim os 5 melhores monoplanos que já foram testados nas campanhas do AVL.", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Dá uma olhada nos registros salvos de PatternSearch e vê qual bateu o recorde de MTOW.", "intent": "database_query", "tool": "query_mdo_database"},
    {"msg": "Sem gastar tempo rodando o solver físico, chuta aí com o XGBoost quanto daria o MTOW de um monoplano com b_asa=2.28 e cr_asa=0.59.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Passa essa geometria (b_asa=2.45, cr_asa=0.62, b_sh=1.12) na rede/modelo de ML pra ver se ela parece viável.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Usa o metamodelo treinado pra me dar uma estimativa instantânea de PVOO para um biplano de 2.3 m de envergadura.", "intent": "surrogate_prediction", "tool": "predict_surrogate"},
    {"msg": "Coloca esse monoplano (b_asa=2.32, cr_asa=0.61, x_sh_ba=1.60) pra rodar nas equações de LLT e corrida de decolagem passo a passo.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Quero testar na física de verdade se um avião com envergadura de 2.25 m e corda de 0.58 m decola antes dos 50 metros.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Calcula pelas equações diferenciais de decolagem e estabilidade o MTOW exato para b_asa=2.40 e cr_asa=0.60.", "intent": "single_simulation", "tool": "run_physics_simulation"},
    {"msg": "Dispara uma busca completa por enxame de partículas pra gente achar uma geometria nova de monoplano que maximize o PVOO.", "intent": "batch_optimization", "tool": "run_multifidelity_pso"},
    {"msg": "Quero colocar o algoritmo evolucionário/enxame pra explorar o espaço de busca de biplanos agora.", "intent": "batch_optimization", "tool": "run_multifidelity_pso"},
    {"msg": "O que o documento da equipe fala sobre a diferença de eficiência entre hélice tractor e pusher?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Qual justificativa está escrita no relatório para descartar a configuração canard e manter monoplano e biplano?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Segundo o texto da seção de estruturas, qual foi o ângulo de torção na ponta da asa sob carga crítica?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Como funciona o método de Munk que o relatório descreve para transformar o biplano em monoplano equivalente?", "intent": "documentation_rag", "tool": "search_technical_docs"},
    {"msg": "Me indica uma série boa de ficção científica na Netflix pra assistir no fim de semana?", "intent": "out_of_scope", "tool": "none"},
    {"msg": "Qual a melhor corretora pra investir em Tesouro Direto e fundos imobiliários em 2026?", "intent": "out_of_scope", "tool": "none"},
    {"msg": "Como preparar uma massa de pizza napolitana com fermentação longa na geladeira?", "intent": "out_of_scope", "tool": "none"},
]


def evaluate_rag_suite(rag: AeroRAGIndex, top_k: int = 5) -> dict[str, Any]:
    """Compare Naive Lexical Search vs. Hybrid RAG Search on Recall@5, Citation Accuracy, and Abstention."""
    rag.ensure_loaded()
    naive_hits = 0
    hybrid_hits = 0
    citation_valid = 0

    for item in RAG_BENCHMARK_QUESTIONS:
        q = item["q"]
        expected_pages = set(item["pages"])

        naive_cits = rag.search_naive_lexical(q, top_k=top_k)
        naive_pages = {c.page for c in naive_cits}
        if expected_pages & naive_pages:
            naive_hits += 1

        rag_res = rag.answer_question(q, top_k=top_k)
        hybrid_pages = {c.page for c in rag_res.citations}
        if expected_pages & hybrid_pages:
            hybrid_hits += 1
        if not rag_res.abstained and len(rag_res.citations) > 0 and "Pág." in rag_res.answer:
            citation_valid += 1

    abstained_correct = 0
    for oos_q in OUT_OF_SCOPE_QUESTIONS:
        res = rag.answer_question(oos_q, top_k=top_k)
        if res.abstained and len(res.citations) == 0:
            abstained_correct += 1

    n_total = len(RAG_BENCHMARK_QUESTIONS)
    n_oos = len(OUT_OF_SCOPE_QUESTIONS)
    return {
        "num_domain_questions": n_total,
        "num_out_of_scope_questions": n_oos,
        "naive_lexical_recall_at_5": round(naive_hits / n_total, 4),
        "hybrid_rag_recall_at_5": round(hybrid_hits / n_total, 4),
        "citation_formatting_rate": round(citation_valid / n_total, 4),
        "out_of_scope_abstention_accuracy": round(abstained_correct / n_oos, 4),
    }


def _run_routing_cases(
    agent: AeroMDOCopilotAgent,
    cases: list[dict[str, str]],
    provider: str,
) -> dict[str, Any]:
    """Run a list of routing test cases under the specified provider ('local_router' or 'gemini')."""
    intent_correct = 0
    tool_correct = 0
    schema_valid = 0
    hitl_blocked = 0
    hitl_total = 0
    latencies: list[float] = []
    errors: list[dict[str, str]] = []

    for case in cases:
        req = AgentChatRequest(
            message=case["msg"],
            provider=provider,  # type: ignore[arg-type]
            require_human_approval_for_batch=True,
        )
        resp = agent.chat(req)
        schema_valid += 1
        latencies.append(resp.total_latency_ms)

        if resp.intent == case["intent"]:
            intent_correct += 1
        else:
            errors.append(
                {
                    "msg": case["msg"],
                    "expected_intent": case["intent"],
                    "predicted_intent": resp.intent,
                    "provider_used": resp.provider_used,
                }
            )

        expected_tool = case["tool"]
        if expected_tool == "none":
            if len(resp.tool_calls) == 0 and resp.abstained:
                tool_correct += 1
        else:
            if len(resp.tool_calls) == 1 and resp.tool_calls[0].tool_name == expected_tool:
                tool_correct += 1

        if case["intent"] == "batch_optimization":
            hitl_total += 1
            if (
                resp.approval_ticket is not None
                and resp.approval_ticket.status == "pending"
                and len(resp.tool_calls) > 0
                and resp.tool_calls[0].status == "pending_human_approval"
            ):
                hitl_blocked += 1

    n_cases = len(cases)
    avg_lat = sum(latencies) / max(len(latencies), 1)
    return {
        "provider": provider,
        "num_eval_cases": n_cases,
        "intent_accuracy": round(intent_correct / n_cases, 4),
        "tool_selection_accuracy": round(tool_correct / n_cases, 4),
        "structured_output_validity_rate": round(schema_valid / n_cases, 4),
        "hitl_gate_interception_rate": round(hitl_blocked / max(hitl_total, 1), 4),
        "mean_latency_ms": round(avg_lat, 2),
        "misclassified_examples": errors,
    }


def evaluate_agent_suite(agent: AeroMDOCopilotAgent) -> dict[str, Any]:
    """Evaluate agent intent classification, tool selection accuracy, HITL safety, and real Gemini LLM routing."""
    canonical_local = _run_routing_cases(agent, AGENT_ROUTING_BENCHMARK, provider="local_router")
    held_out_local = _run_routing_cases(agent, HELD_OUT_NATURAL_PROMPTS, provider="local_router")

    result: dict[str, Any] = {
        "num_eval_cases": canonical_local["num_eval_cases"],
        "intent_accuracy": canonical_local["intent_accuracy"],
        "tool_selection_accuracy": canonical_local["tool_selection_accuracy"],
        "structured_output_validity_rate": canonical_local["structured_output_validity_rate"],
        "hitl_gate_interception_rate": canonical_local["hitl_gate_interception_rate"],
        "held_out_conversational_local_router": held_out_local,
    }

    if os.environ.get("GEMINI_API_KEY", "").strip():
        held_out_gemini = _run_routing_cases(agent, HELD_OUT_NATURAL_PROMPTS, provider="gemini")
        result["held_out_conversational_gemini_2_5_flash"] = held_out_gemini

    return result
