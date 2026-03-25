# metrics/metrics.py

from typing import Dict, Tuple, List
import math
import re
import os
import json
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
import config


try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def _safe_len_ratio(generated: str, reference: str) -> float:
    if not reference:
        return 0.0
    return len(generated) / max(len(reference), 1)


def evaluate_pair(generated: str, reference: str) -> Dict[str, float]:
    """
    Avalia um par (generated, reference) com métricas determinísticas:
    - BERTScore F1
    - ROUGE-L F1
    - razão de tamanho (len_generated / len_reference)
    """
    metrics: Dict[str, float] = {}

    gen = (generated or "").strip()
    ref = (reference or "").strip()

    if not gen or not ref:
        metrics["bertscore_f1"] = 0.0
       # metrics["rougeL_f1"] = 0.0
       # metrics["len_ratio"] = 0.0
        return metrics

    # BERTScore
    if bert_score is not None:
        try:
            # BERTScore mede similaridade semântica.
            # AVISO: Se o texto gerado for repetitivo mas contiver as palavras chave,
            # o score pode ser artificialmente alto.
            P, R, F1 = bert_score(
                [gen],
                [ref],
                lang=config.METRICS_LANGUAGE,
                device="cpu",
                rescale_with_baseline=config.USE_BERTSCORE_BASELINE,
            )
            metrics["bertscore_f1"] = float(F1[0].item())
        except Exception as e:
            print(f"[BERTScore Error] {e}")
            metrics["bertscore_f1"] = 0.0
    else:
        metrics["bertscore_f1"] = 0.0

    # ROUGE-L
    if rouge_scorer is not None:
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(ref, gen)
            metrics["rougeL_f1"] = float(scores["rougeL"].fmeasure)
        except Exception:
            metrics["rougeL_f1"] = 0.0
    else:
        metrics["rougeL_f1"] = 0.0

    metrics["len_ratio"] = _safe_len_ratio(gen, ref)

    return metrics


def metrics_to_prefixed_columns(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Converte {"bertscore_f1": 0.85, ...}
    em {"metric_bertscore_f1": 0.85, ...}
    """
    return {
        f"{config.METRIC_COL_PREFIX}{k}": v
        for k, v in metrics.items()
    }


# =============================================================================
# G-EVAL (LLM-based Metrics)
# =============================================================================

def calculate_g_eval(llm, base_text: str, review_text: str, final_text: str) -> Tuple[float, str, Dict[str, int]]:
    """
    Executa o G-Eval usando o LLM fornecido.
    Retorna (score, justification, usage_dict).
    """
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not config.G_EVAL_ENABLED:
        return 0.0, "G-Eval disabled", empty_usage
        
    base_text = (base_text or "").strip()
    review_text = (review_text or "").strip()
    final_text = (final_text or "").strip()

    if not base_text or not review_text or not final_text:
        return 0.0, "Missing input text", empty_usage

    messages = _build_g_eval_messages(base_text, review_text, final_text)

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        
        # Extrai uso de tokens
        usage = empty_usage.copy()
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if "token_usage" in meta:
                usage = meta["token_usage"]
            elif "usage" in meta:
                usage = meta["usage"]
        
        # Normaliza chaves
        final_usage = {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0)
        }
        if final_usage["total_tokens"] == 0:
            final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]

        score, justification = _parse_g_eval_response(content)
        return score, justification, final_usage
    except Exception as e:
        print(f"[G-Eval Error] {e}")
        return 0.0, f"Error: {str(e)}", empty_usage


def _build_g_eval_messages(base_text: str, review_text: str, final_text: str) -> List:
    system_msg=SystemMessage(content = "You are a strict evaluator of scientific article revisions.")

    human_content=(
        "You are evaluating if a revised paper successfully incorporated the requested changes.\n\n"
        "1. THE REVIEW (Instructions/Critiques from the board):\n"
        f"{review_text}\n\n"
        "2. THE FINAL TEXT (The revised version generated by the AI):\n"
        f"{final_text}\n\n"
        "EVALUATION TASK:\n"
        "Compare the FINAL TEXT against the REVIEW. Did the author implement the requested changes?\n\n"
        "SCORING GUIDE:\n"
        "- 10: All suggestions were perfectly implemented.\n"
        "- 1: The AI completely ignored the review comments.\n\n"
        "Response format:\n"
        "Incorporation of reviews: <score> -- <justification>"
    )

    human_msg=HumanMessage(content = human_content)
    return [system_msg, human_msg]


def _parse_g_eval_response(raw: str) -> Tuple[float, str]:
    raw = raw.strip()
    
    # Tenta encontrar a linha com o padrão esperado
    target_line = None
    for l in raw.splitlines():
        if "Incorporation of reviews" in l:
            target_line = l.strip()
            break
    
    # Se não encontrou a linha específica, tenta procurar o número em todo o texto
    if not target_line:
        # Procura por um número isolado ou no formato esperado em todo o texto
        # Prioriza números entre 1 e 10
        m = re.search(r"\b([1-9]|10)(\.\d+)?\b", raw)
        if m:
            score = float(m.group(0))
            justification = raw
            return score, justification
        else:
            return 0.0, raw

    # Se encontrou a linha, extrai dela
    m = re.search(r"(\d+(\.\d+)?)", target_line)
    score = float(m.group(1)) if m else 0.0
    
    justification = target_line.split("--", 1)[1].strip() if "--" in target_line else raw
    return score, justification


def calculate_g_eval_quality(llm, final_text: str) -> Tuple[float, str, Dict, Dict[str, int]]:
    """
    Executa o G-Eval de Qualidade Textual usando o LLM fornecido.
    Retorna (nota_final, justificativa_completa_json, dict_detalhado, usage_dict).
    """
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not config.G_EVAL_ENABLED:
        return 0.0, "G-Eval disabled", {}, empty_usage

    final_text = (final_text or "").strip()
    if not final_text:
        return 0.0, "Missing input text", {}, empty_usage

    messages = _build_g_eval_quality_messages(final_text)

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        
        # Extrai uso de tokens
        usage = empty_usage.copy()
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if "token_usage" in meta:
                usage = meta["token_usage"]
            elif "usage" in meta:
                usage = meta["usage"]
        
        # Normaliza chaves
        final_usage = {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0)
        }
        if final_usage["total_tokens"] == 0:
            final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]

        nota, just, data = _parse_g_eval_quality_response(content)
        return nota, just, data, final_usage
    except Exception as e:
        print(f"[G-Eval Quality Error] {e}")
        return 0.0, f"Error: {str(e)}", {}, empty_usage


def _build_g_eval_quality_messages(final_text: str) -> List:
    prompt = f"""You are a strict evaluator of text quality.
Your task is to evaluate an AI-GENERATED TEXT based on linguistic and discursive criteria.

IMPORTANT
- Be critical and rigorous.
- Avoid generosity bias: very high scores should be rare.
- Do not rewrite or correct the text.
- Do not include any text outside the final JSON.

SCORE SCALE (1 to 10)

1–2 = Very poor, serious flaws
3–4 = Poor, many problems
5–6 = Fair, acceptable but with several issues
7–8 = Good, few problems
9 = Excellent, very rare issues
10 = Perfect, practically impossible for AI texts (extremely rare use)

CRITERIA (assign an integer score from 1 to 10 and a short justification of max 2 sentences)

1. Global Coherence (weight 0.25)
Consistency of meaning, absence of contradictions, and logical progression.

2. Textual Cohesion (weight 0.20)
Connectivity between sentences, appropriate use of connectives and references.

3. Clarity and Readability (weight 0.20)
Ease of understanding, objectivity, and absence of ambiguities.

4. Terminological Precision and Domain Adequacy (weight 0.20)
Correct and consistent use of technical terms and appropriate register.

5. Structure and Organization (weight 0.15)
Logical order of ideas and text organization.

FINAL SCORE CALCULATION (mandatory)

Calculate EXACTLY:

final_score =
(coherence * 0.25) +
(cohesion * 0.20) +
(clarity * 0.20) +
(precision * 0.20) +
(structure * 0.15)

Round to 1 decimal place.

TEXT TO EVALUATE:
<<<
{final_text}
>>>

OUTPUT FORMAT (return ONLY valid JSON)

{{
  "coherence": {{"score": X, "justification": "..."}},
  "cohesion": {{"score": X, "justification": "..."}},
  "clarity": {{"score": X, "justification": "..."}},
  "precision": {{"score": X, "justification": "..."}},
  "structure": {{"score": X, "justification": "..."}},
  "final_score": Y
}}"""

    return [HumanMessage(content=prompt)]


def _parse_g_eval_quality_response(raw: str) -> Tuple[float, str, Dict]:
    data = {}
    # 1. Tentativa via JSON
    try:
        json_str = raw
        if "```json" in raw:
            json_str = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            json_str = raw.split("```")[1].split("```")[0]
        else:
            # Tenta encontrar o início e fim do JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_str = raw[start:end+1]
            else:
                json_str = raw
        
        json_str = json_str.strip()
        data = json.loads(json_str)
        
        # Tenta pegar final_score (ou nota_final como fallback)
        final_score = float(data.get("final_score", data.get("nota_final", 0.0)))
        
        # Se achou nota > 0, retorna
        if final_score > 0:
            justificativa = json.dumps(data, ensure_ascii=False)
            return final_score, justificativa, data
            
    except Exception:
        # Falha silenciosa no JSON, tenta regex
        pass

    # 2. Fallback via Regex (procura padrões comuns)
    # Padrão: "final_score": 8.5
    m = re.search(r'"?final_score"?\s*:\s*(\d+(\.\d+)?)', raw, re.IGNORECASE)
    if m:
        return float(m.group(1)), raw, {}
        
    # Padrão: final_score = 8.5
    m = re.search(r'final_score\s*=\s*(\d+(\.\d+)?)', raw, re.IGNORECASE)
    if m:
        return float(m.group(1)), raw, {}

    # Padrão: "nota_final": 8.5 (caso o modelo responda em pt)
    m = re.search(r'"?nota_final"?\s*[:=]\s*(\d+(\.\d+)?)', raw, re.IGNORECASE)
    if m:
        return float(m.group(1)), raw, {}

    print(f"[G-Eval Quality Parse Error] Could not extract score. Raw output start: {raw[:100]}...")
    return 0.0, raw, {}


def save_comparative_txt(agent_name: str, model_name: str, content: str):
    """
    Cria ou atualiza um arquivo TXT para cada combinação de agente e modelo.
    Exemplo de saída: comparison_results/planner_deepseek-v3.txt
    """
    output_dir="comparison_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clean_model_name=model_name.replace(":", "_").replace("/", "_")
    file_name=f"{agent_name}_{clean_model_name}.txt"
    file_path=os.path.join(output_dir, file_name)

    with open(file_path, "a", encoding = "utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"TEST DATE: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"AGENT: {agent_name.upper()}\n")
        f.write(f"MODEL USED: {model_name}\n")
        f.write(f"{'-' * 70}\n")
        f.write(f"RESPONSE:\n\n{content}\n")
        f.write(f"{'=' * 70}\n\n")
