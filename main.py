# main.py

import logging
import config
import os
import subprocess
import json
from agents.graph_pipeline import DRFTGraphPipeline
from services.benchmark_baseline import run_dataset_baseline
from metrics.metrics import save_comparative_txt

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("DRFT_MAIN")

def pull_ollama_model(logger, model_name):
    """
    Baixa o modelo no Ollama antes de executar.
    Usa curl conforme solicitado.
    """
    logger.info(f"Baixando modelo {model_name} no Ollama...")
    
    url = "http://10.93.48.97:11435/api/pull"
    
    cmd = [
        "curl",
        "-X", "POST",
        url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"model": model_name})
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Modelo {model_name} baixado (ou já existente) com sucesso.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao baixar modelo {model_name}: {e}")

def print_metrics(logger, title, metrics):
    if metrics:
        logger.info(f"\n{title}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    else:
        logger.info(f"Nenhuma métrica disponível para {title}.")

def compare_metrics(logger, agent_means, dataset_means):
    if agent_means and dataset_means:
        logger.info("\n===== COMPARAÇÃO AGENTE vs DATASET GOLD =====")
        all_keys = sorted(set(agent_means.keys()).union(dataset_means.keys()))
        for k in all_keys:
            a = agent_means.get(k, float("nan"))
            d = dataset_means.get(k, float("nan"))
            logger.info(f"{k}: agente={a:.4f} | dataset={d:.4f}")

def main():
    logger = setup_logging()

    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Download de modelos
    for current_model in config.MODELS_TO_COMPARE:
        logger.info(f"\n{'=' * 40}\nDOWNLOADING MODEL: {current_model}\n{'=' * 40}")

        if not current_model.startswith("gpt-5") and not current_model.startswith("gpt-4"):
            pull_ollama_model(logger, current_model)

    # Loop de modelos
    for current_model in config.MODELS_TO_COMPARE:
        logger.info(f"\n{'=' * 40}\nRUNNING EXPERIMENT WITH: {current_model}\n{'=' * 40}")

        clean_model_name = current_model.replace(":", "_").replace("-", "_").replace("/", "_")
        config.OUTPUT_CSV = os.path.join(output_dir, f"results_{clean_model_name}.csv")

        # Configura o modelo atual para todos os agentes
        config.PLANNER_MODEL = current_model
        config.WRITER_MODEL = current_model
        config.LLM_MODEL_NAME = current_model
        
        # Opcional: Se quiser que o Judge use o mesmo modelo, descomente abaixo.
        # Caso contrário, ele usará o que estiver em config.G_EVAL_MODEL (padrão gpt-4o-mini)
        # config.G_EVAL_MODEL = current_model

        # 1) Roda o MAS (multiagente) para o modelo atual
        logger.info(f"Starting DRFT pipeline for model: {current_model}...")
        pipeline = DRFTGraphPipeline()

        # Aqui o pipeline roda.
        agent_means = pipeline.run()

        print(f"\nEXPERIMENT COMPLETED: {current_model}")
        print(f"CSV Generated: {config.OUTPUT_CSV}")

        print_metrics(logger, f"OVERALL MEANS - MODEL: {current_model}", agent_means)

    # 2) Roda o baseline do DATASET GOLD
    logger.info("\nStarting GOLD DATASET baseline...")
    dataset_means = run_dataset_baseline()
    print_metrics(logger, "Final Summary - GOLD DATASET Score:", dataset_means)

    # 3) Comparação
    compare_metrics(logger, agent_means, dataset_means)

if __name__ == "__main__":
    main()