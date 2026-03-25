# services/llm_factory.py

from langchain_openai import ChatOpenAI
import config

class LLMFactory:
    """
    Factory para criar instâncias de LLM (OpenAI ou vLLM local).
    """

    @staticmethod
    def create_llm(model_name: str, temperature: float) -> ChatOpenAI:
        """
        Cria um LLM com base na configuração (OpenAI ou Local).
        """
        if model_name.startswith("gpt"):
            return LLMFactory._get_openai_llm(model_name, temperature)
        else:
            return LLMFactory._get_local_vllm(model_name, temperature)

    @staticmethod
    def _get_openai_llm(model_name: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=temperature,
            timeout = 600.0,
        )

    @staticmethod
    def _get_local_vllm(model_name: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            base_url=config.CHAT_MODEL_URL,
            api_key="nao_importa",
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=temperature,
            timeout = 900.0,
        )

def get_planner_llm() -> ChatOpenAI:
    return LLMFactory.create_llm(
        model_name=config.PLANNER_MODEL,
        temperature=config.LLM_TEMPERATURE_PLANNER
    )

def get_writer_llm() -> ChatOpenAI:
    return LLMFactory.create_llm(
        model_name=config.WRITER_MODEL,
        temperature=config.LLM_TEMPERATURE_WRITER
    )

def get_judge_llm() -> ChatOpenAI:
    # O Judge geralmente usa um modelo forte (GPT-4) ou o mesmo modelo local.
    # Aqui usamos a variável G_EVAL_MODEL.
    return LLMFactory.create_llm(
        model_name=config.G_EVAL_MODEL,
        temperature=config.G_EVAL_TEMPERATURE
    )
