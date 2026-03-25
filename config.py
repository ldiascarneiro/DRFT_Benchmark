import os

# ==========================
# ARQUIVOS
# ==========================

# CSV principal do DRFT já cortado/ajustado (delimitador "#")
# INPUT_CSV = os.getenv("DRFT_INPUT_CSV", "data/DRFT_dataset-cut.csv")
INPUT_CSV = os.getenv("DRFT_INPUT_CSV", "data/DRFT_dataset.csv")
OUTPUT_CSV = os.getenv("DRFT_OUTPUT_CSV", "data/DRFT_multiagent_output.csv")

# Número máximo de exemplos (linhas) a processar
TOP_K_EXAMPLES = os.getenv("DRFT_TOP_K")
TOP_K_EXAMPLES = int(TOP_K_EXAMPLES) if TOP_K_EXAMPLES not in (None, "", "None") else None

# ==========================
# MODE: LOCAL (vLLM) ou OPENAI
# ==========================

# Chave OpenAI (NÃO deixe hardcoded em produção; use variável de ambiente)
os.environ["OPENAI_API_KEY"] = "API_KEY_AQUI"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Modelos para cada agente (serão sobrescritos pelo loop em main.py)
PLANNER_MODEL = os.getenv("DRFT_PLANNER_MODEL", "gpt-4o-mini")
WRITER_MODEL = os.getenv("DRFT_WRITER_MODEL", "gpt-4o-mini")
G_EVAL_MODEL = os.getenv("DRFT_G_EVAL_MODEL", "gpt-4o-mini")

# ==========================
# MODELO (vLLM LOCAL)
# ==========================

# Endpoint do servidor vLLM para inferência
CHAT_MODEL_URL = os.getenv("DRFT_CHAT_MODEL_URL", "IP_SERVIDOR")

# URL base para operações de API do Ollama (ex: /api/pull)
OLLAMA_API_URL = os.getenv("DRFT_OLLAMA_API_URL", "IP_SERVIDOR")

# Nome do modelo a ser usado se não especificado de outra forma (fallback)
LLM_MODEL_NAME = os.getenv("DRFT_LLM_MODEL_NAME", "qwen2.5:32b")

# Limite de tokens gerados (max_tokens do ChatOpenAI)
LLM_MAX_TOKENS = int(os.getenv("DRFT_LLM_MAX_TOKENS", "4096"))

# Temperaturas
LLM_TEMPERATURE_PLANNER = float(os.getenv("DRFT_LLM_TEMP_PLANNER", "0.2"))
LLM_TEMPERATURE_WRITER = float(os.getenv("DRFT_LLM_TEMP_WRITER", "0.3"))

# ==========================
# LIMITES DE CONTEXTO (SEGURANÇA)
# ==========================

# Limite de caracteres do draft e dos reviews passados ao modelo
MAX_CHARS_BASE = int(os.getenv("DRFT_MAX_CHARS_BASE", "120000"))
MAX_CHARS_REVIEW = int(os.getenv("DRFT_MAX_CHARS_REVIEW", "60000"))

# ==========================
# COLUNAS DO DATASET
# ==========================
COL_IDX = "idx"
COL_NAME = "name"
COL_TITLE = "title"
COL_BASE_PAPER = "base_text"
COL_REVIEW = "review"
COL_ARXIV_PAPER = "final_text"
COL_REVISION_PLAN = "revision_plan"
COL_GENERATED_PAPER = "generated_paper"
COL_BERTSCORE_GOLD = "bertscore"
COL_G_EVAL_GOLD = "g_eval_score"
COL_G_EVAL_JUST_GOLD = "g_eval_justification"
METRIC_COL_PREFIX = "metric_"

# ==========================
# LOGGING
# ==========================

LOG_LEVEL = os.getenv("DRFT_LOG_LEVEL", "INFO")

# ==========================
# AVALIAÇÃO (BERTScore / ROUGE)
# ==========================

METRICS_LANGUAGE = os.getenv("DRFT_METRICS_LANG", "en")
USE_BERTSCORE_BASELINE = os.getenv("DRFT_BERT_BASELINE", "1") not in ("0", "False", "false")

# ==========================
# G-EVAL (LLM-AS-A-JUDGE)
# ==========================

G_EVAL_ENABLED = os.getenv("DRFT_G_EVAL_ENABLED", "1") not in ("0", "False", "false")
G_EVAL_TEMPERATURE = float(os.getenv("DRFT_G_EVAL_TEMP", "0.0"))

# ==========================
# MODELS TO COMPARE
# ==========================
MODELS_TO_COMPARE = [
"deepseek-r1:8b",
"lfm2.5-thinking:1.2b",
"phi4-mini:3.8b",
"qwen3:8b",
"falcon3:10b",

"qwen3:14b",
"gemma3:12b",
"deepseek-r1:14b",
"phi4:14b",
##"gpt-oss:20b"

"mistral-small3.2:24b",
"nemotron-3-nano:30b",
"olmo-3.1:32b",
"qwen3:32b",
"gemma3:27b",
"deepseek-r1:32b",
"glm-4.7-flash:q8_0",

"qwen3-next:80b",
"deepseek-r1:70b",
"llama3.3:70b",
"llama4:16x17b"  ,       #67b
"glm-4.7-flash:bf16" ,   #60b
]
