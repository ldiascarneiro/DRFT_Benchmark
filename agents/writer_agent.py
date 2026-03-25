# agents/writer_agent.py

from typing import Optional, List, Tuple, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    """
    WRITER AGENT (ENGLISH VERSION, SECTION-BY-SECTION, CONSERVATIVE)

    Este agente recebe:
        - o draft original (base_paper),
        - o plano de revisão estruturado (revision_plan),
        - o título e um ID do artigo.

    Objetivo:
        - Revisar o artigo SEÇÃO A SEÇÃO, com base no plano,
        - Preservando ao máximo o texto original (estrutura, seções e frases),
        - Editando apenas o que for necessário para responder às críticas dos revisores.
    """

    def build_messages(
        self,
        title: str,
        base_paper: str,
        revision_plan: str,
        name: Optional[str] = "",
    ) -> List:
        title = (title or "").strip()
        base_paper = (base_paper or "").strip()
        revision_plan = (revision_plan or "").strip()
        name = (name or "").strip()

        # Truncamento defensivo
        base_paper = self._truncate_if_needed(base_paper, "MAX_CHARS_BASE")

        id_info = f"Paper ID: {name}\n" if name else ""
        title_info = f"Title: {title}\n" if title else ""

        system_content = (
            "You are an expert scientific writing assistant acting as a WRITER for an academic article.\n\n"
            "Your task is to revise the ORIGINAL DRAFT using a SECTION-BY-SECTION revision plan.\n"
            "Follow these STRICT rules:\n"
            "1. Preserve the overall structure and section ordering of the original draft.\n"
            "   - Keep the same section headings.\n"
            "   - Do NOT introduce new sections unless the revision plan explicitly requires this.\n"
            "2. Work SECTION BY SECTION:\n"
            "   - For each [SECTION: ...] block in the revision plan, find the corresponding section in the draft.\n"
            "   - Apply the 'Revision actions' to that section.\n"
            "3. Be CONSERVATIVE:\n"
            "   - Prefer editing existing sentences instead of rewriting them completely.\n"
            "   - Only change what is necessary to address reviewer comments, improve clarity, or fix errors.\n"
            "   - Do NOT change correct technical content or notation.\n"
            "4. Do NOT add your own opinions or commentary.\n"
            "5. The final output must be a SINGLE, COMPLETE revised article, with the same section structure as "
            "the original draft (unless the plan explicitly says otherwise).\n"
            "6. Do NOT include explanations of the changes. Output only the revised article.\n"
        )

        human_content = (
            f"{id_info}{title_info}"
            "ORIGINAL DRAFT (to be revised):\n"
            "----------------------------------------\n"
            f"{base_paper}\n\n"
            "REVISION PLAN (organized by sections):\n"
            "----------------------------------------\n"
            f"{revision_plan}\n\n"
            "TASK:\n"
            "Using the revision plan, revise the draft SECTION BY SECTION, making the minimal changes required to "
            "address the issues and actions described in the plan. Preserve the structure and as much of the original "
            "wording as possible. Return ONLY the fully revised article.\n"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
        return messages

    def write(
        self,
        title: str,
        base_paper: str,
        revision_plan: str,
        name: Optional[str] = "",
    ) -> Tuple[str, Dict[str, int]]:
        if not base_paper or not revision_plan:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        messages = self.build_messages(
            title=title,
            base_paper=base_paper,
            revision_plan=revision_plan,
            name=name,
        )

        return self.invoke(messages)
