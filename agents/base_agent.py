from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple, Dict
from langchain_core.messages import BaseMessage
import config

class BaseAgent(ABC):
    """
    Classe base para agentes do sistema.
    Fornece utilitários comuns como truncamento de texto e estrutura básica.
    """

    def __init__(self, llm):
        self.llm = llm

    def _truncate_if_needed(self, text: str, max_chars_attr: str) -> str:
        """
        Corta o texto se houver um limite definido em config.
        """
        max_chars = getattr(config, max_chars_attr, None)
        if max_chars is None:
            return text
        try:
            max_chars = int(max_chars)
        except Exception:
            return text
        if max_chars <= 0:
            return text
        return text[:max_chars]

    @abstractmethod
    def build_messages(self, **kwargs) -> List[BaseMessage]:
        """
        Constrói a lista de mensagens para o LLM.
        Deve ser implementado pelas subclasses.
        """
        pass

    def invoke(self, messages: List[BaseMessage]) -> Tuple[str, Dict[str, int]]:
        """
        Invoca o LLM com as mensagens fornecidas e retorna o conteúdo da resposta
        E um dicionário com o uso de tokens (prompt_tokens, completion_tokens, total_tokens).
        """
        response = self.llm.invoke(messages)
        content = getattr(response, "content", str(response))
        
        # Tenta extrair metadados de uso. 
        # A estrutura exata depende do provedor (OpenAI vs vLLM/LangChain),
        # mas geralmente está em response.response_metadata['token_usage']
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if "token_usage" in meta:
                usage = meta["token_usage"]
            elif "usage" in meta: # Algumas implementações usam 'usage'
                usage = meta["usage"]
        
        # Normaliza chaves se necessário (alguns provedores podem usar camelCase)
        # Garante que temos as chaves padrão
        final_usage = {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        # Recalcula total se estiver zerado mas tivermos as partes
        if final_usage["total_tokens"] == 0:
            final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]

        return content, final_usage
