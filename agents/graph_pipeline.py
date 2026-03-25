import time
import logging
from typing import TypedDict, Dict, List

import pandas as pd
from langgraph.graph import StateGraph, END

import config
from services.llm_factory import get_planner_llm, get_writer_llm, get_judge_llm
from agents.planner_agent import PlannerAgent
from agents.writer_agent import WriterAgent
from agents.evaluator_agent import EvaluatorAgent
from metrics.metrics import evaluate_pair, metrics_to_prefixed_columns, save_comparative_txt


logger = logging.getLogger("DRFT_GRAPH")


class DRFTState(TypedDict, total=False):
    """
    Estado compartilhado no LangGraph.
    """
    idx: int
    name: str
    title: str
    base_paper: str
    review: str
    arxiv_paper: str

    revision_plan: str
    generated_paper: str

    metrics: Dict[str, float]
    g_eval_justification: str
    g_eval_quality_justification: str
    
    # Contagem de tokens acumulada
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class DRFTGraphPipeline:
    """
    Sistema Multiagente (MAS) via LangGraph.
    """

    def __init__(self):
        logger.info("Instanciando LLMs e agentes...")
        self.planner_agent = PlannerAgent(get_planner_llm())
        self.writer_agent = WriterAgent(get_writer_llm())
        self.evaluator_agent = EvaluatorAgent(get_judge_llm())

        self.graph = self._build_graph()

    def _update_token_usage(self, state: DRFTState, usage: Dict[str, int]) -> DRFTState:
        """
        Atualiza os contadores de tokens no estado.
        """
        state["total_prompt_tokens"] = state.get("total_prompt_tokens", 0) + usage.get("prompt_tokens", 0)
        state["total_completion_tokens"] = state.get("total_completion_tokens", 0) + usage.get("completion_tokens", 0)
        state["total_tokens"] = state.get("total_tokens", 0) + usage.get("total_tokens", 0)
        return state

    def _planner_node(self, state: DRFTState) -> DRFTState:
        start_t = time.perf_counter()
        logger.info(f"[Planner] Iniciando para idx={state.get('idx')}")
        
        plan, usage = self.planner_agent.generate_plan(
            title=state.get("title", "") or "",
            base_paper=state.get("base_paper", "") or "",
            review=state.get("review", "") or "",
            name=state.get("name", "") or "",
        )
        
        elapsed = time.perf_counter() - start_t
        self._log_agent_completion("Planner", state.get("idx"), plan, usage, elapsed)
        
        state["revision_plan"] = plan
        return self._update_token_usage(state, usage)

    def _writer_node(self, state: DRFTState) -> DRFTState:
        start_t = time.perf_counter()
        logger.info(f"[Writer] Iniciando para idx={state.get('idx')}")
        
        paper, usage = self.writer_agent.write(
            title=state.get("title", "") or "",
            base_paper=state.get("base_paper", "") or "",
            revision_plan=state.get("revision_plan", "") or "",
            name=state.get("name", "") or "",
        )
        
        elapsed = time.perf_counter() - start_t
        self._log_agent_completion("Writer", state.get("idx"), paper, usage, elapsed)
        
        state["generated_paper"] = paper
        return self._update_token_usage(state, usage)

    def _evaluator_node(self, state: DRFTState) -> DRFTState:
        start_t=time.perf_counter()
        logger.info(f"[Evaluator] Iniciando para idx={state.get('idx')}")

        gen=state.get("generated_paper", "") or ""
        draft=state.get("base_paper", "") or ""
        reviews=state.get("review", "") or ""
        reference=state.get("arxiv_paper", "") or ""

        metrics, g_just, g_quality_just, usage = self.evaluator_agent.evaluate(
            generated_paper = gen,
            base_paper = draft,
            review = reviews,
            reference_paper = reference
        )

        elapsed=time.perf_counter() - start_t

        print(f"\n--- ARTIGO FINALIZADO (idx={state.get('idx')}) ---")
        print(f"BERTScore (Simil. Draft): {metrics.get('bertscore_f1', 0):.4f}")

        # Pega o score do G-Eval (ajuste a chave se o evaluator usar outro nome)
        g_score=metrics.get('g_eval_score', 0)
        g_quality_score=metrics.get('g_eval_quality_score', 0)
        print(f"G-Eval (Aderência Banca): {g_score}")
        print(f"G-Eval (Qualidade Textual): {g_quality_score}")
        print(f"Justificativa Aderência: {g_just[:150]}...")
        print(f"Justificativa Qualidade: {g_quality_just[:150]}...")
        print(f"Tokens Usados (Evaluator): {usage.get('total_tokens', 0)}")
        print(f"{'-' * 40}")

        logger.info(
            f"[Evaluator] Concluído para idx={state.get('idx')} em {elapsed:.2f}s "
            f"(metrics={metrics}, tokens={usage.get('total_tokens', 0)})"
        )

        state["metrics"]=metrics
        state["g_eval_justification"]=g_just
        state["g_eval_quality_justification"]=g_quality_just
        
        return self._update_token_usage(state, usage)

    def _log_agent_completion(self, agent_name: str, idx: int, result: str, usage: Dict[str, int], elapsed: float):
        logger.info(
            f"[{agent_name}] Concluído para idx={idx} em {elapsed:.2f}s. "
            f"Output len={len(result or '')}. "
            f"Tokens: {usage.get('total_tokens', 0)} (P:{usage.get('prompt_tokens', 0)} + C:{usage.get('completion_tokens', 0)})"
        )

        save_comparative_txt(
            agent_name = agent_name.lower(),  # Salva como 'planner' ou 'writer'
            model_name = config.PLANNER_MODEL,  # Pega o modelo que o main.py injetou
            content = result if result else "Sem conteúdo gerado."
        )

    def _build_graph(self):
        workflow = StateGraph(DRFTState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("evaluator", self._evaluator_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "writer")
        workflow.add_edge("writer", "evaluator")
        workflow.add_edge("evaluator", END)

        logger.info("Compilando grafo LangGraph...")
        return workflow.compile()

    def run(self):
        logger.info(f"Carregando dataset: {config.INPUT_CSV}")
        df = pd.read_csv(config.INPUT_CSV, sep="#")

        if config.COL_IDX not in df.columns:
            df[config.COL_IDX] = range(len(df))

        if config.TOP_K_EXAMPLES is not None:
            df = df.head(config.TOP_K_EXAMPLES).copy()
            logger.info(f"Processando apenas TOP_K_EXAMPLES={config.TOP_K_EXAMPLES}")

        results = []
        tempos_totais = []
        for idx, row in df.iterrows():
            logger.info(
                f"\n=== Processando linha idx={idx} "
                f"name={row.get(config.COL_NAME, '')} "
                f"title={row.get(config.COL_TITLE, '')} ==="
            )
            
            initial_state = self._build_initial_state_from_row(idx, row)
            
            # Timer total
            start_linha = time.perf_counter()
            try:
                final_state = self.graph.invoke(initial_state)
            except Exception as e:
                logger.error(f"Erro ao processar idx={idx}: {e}")
                final_state = {}
            elapsed_linha = time.perf_counter() - start_linha
            tempos_totais.append(elapsed_linha)

            results.append(final_state)

        return self._process_results(df, results, tempos_totais)

    def _build_initial_state_from_row(self, idx: int, row: pd.Series) -> DRFTState:
        return DRFTState(
            idx=int(idx),
            name=str(row.get(config.COL_NAME, "")) if config.COL_NAME in row else "",
            title=str(row.get(config.COL_TITLE, "")) if config.COL_TITLE in row else "",
            base_paper=str(row.get(config.COL_BASE_PAPER, "")),
            review=str(row.get(config.COL_REVIEW, "")),
            arxiv_paper=str(row.get(config.COL_ARXIV_PAPER, "")),
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0
        )

    def _process_results(self, df: pd.DataFrame, results: List[Dict], tempos_totais: List[float]) -> Dict[str, float]:
        revision_plans = []
        generated_papers = []
        g_eval_justs = []
        g_eval_quality_justs = []
        
        # Listas para tokens
        prompt_tokens_list = []
        completion_tokens_list = []
        total_tokens_list = []
        
        all_metrics = {}

        for res in results:
            revision_plans.append(res.get("revision_plan", ""))
            generated_papers.append(res.get("generated_paper", ""))
            g_eval_justs.append(res.get("g_eval_justification", ""))
            g_eval_quality_justs.append(res.get("g_eval_quality_justification", ""))
            
            prompt_tokens_list.append(res.get("total_prompt_tokens", 0))
            completion_tokens_list.append(res.get("total_completion_tokens", 0))
            total_tokens_list.append(res.get("total_tokens", 0))
            
            metrics = res.get("metrics", {}) or {}
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(float(v))

        df[config.COL_REVISION_PLAN] = revision_plans
        df[config.COL_GENERATED_PAPER] = generated_papers
        df["g_eval_justification"] = g_eval_justs
        df["g_eval_quality_justification"] = g_eval_quality_justs
        
        # Colunas de tokens
        df["total_prompt_tokens"] = prompt_tokens_list
        df["total_completion_tokens"] = completion_tokens_list
        df["total_tokens"] = total_tokens_list
        
        # Adiciona a coluna de tempo total ao DataFrame
        while len(tempos_totais) < len(df):
            tempos_totais.append(0.0)
        df["tempo_total"] = tempos_totais

        if all_metrics:
            self._add_metrics_to_df(df, all_metrics)

        # Salva o CSV individual do modelo
        logger.info(f"Saving results to: {config.OUTPUT_CSV}")
        df.to_csv(config.OUTPUT_CSV, sep="#", index=False)

        # Calcula as medias finais do modelo
        final_means = {k: sum(v)/len(v) for k, v in all_metrics.items() if v}
        
        # Calcula somatório e média dos tempos totais
        total_tempo_sum = sum(tempos_totais)
        total_tempo_mean = total_tempo_sum / len(tempos_totais) if tempos_totais else 0
        
        # Médias de tokens
        avg_tokens = sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else 0

        print(f"\n" + "=" * 50)
        print(f"OVERALL MODEL SUMMARY: {config.PLANNER_MODEL}")
        print(f"Average BERTScore: {final_means.get('bertscore_f1', 0):.4f}")

        # Busca a chave do G-Eval (tentando as duas variações comuns)
        g_score = final_means.get('g_eval_score', final_means.get('g_eval', 0))
        g_quality_score = final_means.get('g_eval_quality_score', 0)
        print(f"Average G-Eval (Aderência): {g_score:.2f}")
        print(f"Average G-Eval (Qualidade): {g_quality_score:.2f}")
        print(f"Average Total Time per Line: {total_tempo_mean:.2f}s")
        print(f"Average Total Tokens per Line: {avg_tokens:.1f}")
        print(f"Total Time for Model: {total_tempo_sum:.2f}s")
        print("=" * 50 + "\n")

        return final_means

    def _add_metrics_to_df(self, df: pd.DataFrame, all_metrics: Dict[str, List[float]]):
        # Inicializa colunas
        sample_metrics = {k: 0.0 for k in all_metrics.keys()}
        for col in metrics_to_prefixed_columns(sample_metrics).keys():
            if col not in df.columns:
                df[col] = 0.0

        # Preenche valores
        for i in range(len(df)):
            row_metrics = {}
            for k, values in all_metrics.items():
                if i < len(values):
                    row_metrics[k] = values[i]
            
            prefixed = metrics_to_prefixed_columns(row_metrics)
            for col_name, value in prefixed.items():
                df.at[df.index[i], col_name] = value
