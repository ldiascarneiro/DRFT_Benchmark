import logging
from typing import Dict, Tuple

from agents.base_agent import BaseAgent
from metrics.metrics import evaluate_pair, calculate_g_eval, calculate_g_eval_quality

logger = logging.getLogger("DRFT_EVALUATOR")

class EvaluatorAgent(BaseAgent):
    """
    Agente AVALIADOR (EvaluatorAgent)
    
    Responsável por orquestrar a avaliação:
    1. Calcular métricas determinísticas (BERTScore, ROUGE, etc.)
    2. Invocar o cálculo do G-Eval (que agora reside em metrics.py)
    """

    def build_messages(self, **kwargs):
        # Este agente agora delega a construção de mensagens para metrics.py
        # ou não usa mensagens diretamente se apenas orquestrar chamadas.
        # Mantido para compatibilidade com BaseAgent se necessário.
        pass

    def evaluate(self, generated_paper: str, base_paper: str, review: str, reference_paper: str = None) -> Tuple[Dict[str, float], str, str, Dict[str, int]]:
        """
        Executa a avaliação completa.
        Retorna: (dicionário de métricas, justificativa do G-Eval Aderência, justificativa do G-Eval Qualidade, usage_dict)
        """
        # 1. Métricas clássicas (determinísticas)
        # Se houver um texto de referência (Gold), usamos ele.
        # Caso contrário, usamos o base_paper (o que pode não ser ideal, mas serve de fallback).
        ref_text = reference_paper if reference_paper else base_paper
        metrics = evaluate_pair(generated_paper, ref_text)

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 2. G-Eval (LLM-as-a-Judge) - Aderência à revisão
        # Passamos o self.llm para a função de métrica, pois ela precisa de um LLM para rodar.
        g_score, g_just, usage_g1 = calculate_g_eval(
            llm=self.llm,
            base_text=base_paper,
            review_text=review,
            final_text=generated_paper
        )
        
        metrics["g_eval_score"] = g_score
        
        # Acumula uso
        for k in total_usage:
            total_usage[k] += usage_g1.get(k, 0)

        # 3. G-Eval (LLM-as-a-Judge) - Qualidade Textual
        g_quality_score, g_quality_just, _, usage_g2 = calculate_g_eval_quality(
            llm=self.llm,
            final_text=generated_paper
        )

        metrics["g_eval_quality_score"] = g_quality_score
        
        # Acumula uso
        for k in total_usage:
            total_usage[k] += usage_g2.get(k, 0)
        
        return metrics, g_just, g_quality_just, total_usage
