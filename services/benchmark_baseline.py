# Created by Leandro Carneiro at 21/11/2025
# Description:
# -----------------------------------------------------------
# benchmark_baseline.py

import logging
from typing import Dict, List

import pandas as pd

import config
from metrics.metrics import evaluate_pair

logger = logging.getLogger("DRFT_DATASET")


def run_dataset_baseline() -> Dict[str, float]:
    """
    Calcula as MÉDIAS GERAIS das métricas para o DATASET GOLD (versões finais humanas),
    usando o próprio CSV DRFT_dataset-cut.csv como fonte de verdade.

    Regras:
      - BERTScore: usa diretamente a coluna `bertscore` do CSV (baseline gold).
      - g_eval_score: usa diretamente a coluna `g_eval_score` do CSV.
      - rougeL_f1 e len_ratio: são calculados via evaluate_pair(final_text, base_text)
        apenas para contextualizar o baseline (não são gold pré-computados).
    """
    logger.info(f"Carregando dataset gold para baseline: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV, sep="#")

    if config.TOP_K_EXAMPLES is not None:
        df = df.head(config.TOP_K_EXAMPLES).copy()
        logger.info(f"[DATASET] Usando apenas TOP_K_EXAMPLES={config.TOP_K_EXAMPLES}")

    all_metrics: Dict[str, List[float]] = {}

    for idx, row in df.iterrows():
        base = str(row.get(config.COL_BASE_PAPER, "") or "")
        final = str(row.get(config.COL_ARXIV_PAPER, "") or "")

        if not base or not final:
            logger.warning(f"[DATASET] Pulando idx={idx}: base_text ou final_text vazio.")
            continue

        title = str(row.get(config.COL_TITLE, "") or "")
        logger.info(f"[DATASET] Avaliando exemplo idx={idx} title={title}")

        # 1) Métricas clássicas: FINAL HUMANO vs DRAFT (para ROUGE/len_ratio).
        # Para o baseline, comparamos o final humano com o draft original.
        metrics = evaluate_pair(final, base)

        # 2) BERTScore GOLD (pré-computado no CSV).
        try:
            bs_gold = float(row.get(config.COL_BERTSCORE_GOLD, 0.0) or 0.0)
        except ValueError:
            bs_gold = 0.0
        metrics["bertscore_f1"] = bs_gold

        # 3) G-Eval GOLD (pré-computado no CSV).
        try:
            ge_gold = float(row.get(config.COL_G_EVAL_GOLD, 0.0) or 0.0)
        except ValueError:
            ge_gold = 0.0
        metrics["g_eval_score"] = ge_gold

        logger.info(
            f"[DATASET] Exemplo idx={idx} "
            f"(bertscore_f1={metrics['bertscore_f1']:.4f}, "
            f"rougeL_f1={metrics['rougeL_f1']:.4f}, "
            f"len_ratio={metrics['len_ratio']:.4f}, "
            f"g_eval_score={metrics['g_eval_score']:.2f})"
        )

        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(float(v))

    global_means: Dict[str, float] = {}
    for k, values in all_metrics.items():
        if not values:
            continue
        global_means[k] = sum(values) / len(values)

    # NÃO imprime o resumo aqui, para evitar duplicidade com o DRFT_MAIN.
    if not global_means:
        logger.info("[DATASET] Nenhuma métrica calculada no baseline.")

    return global_means
