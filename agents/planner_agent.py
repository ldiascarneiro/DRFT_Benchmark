# agents/planner_agent.py

from typing import Optional, List, Tuple, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """
    Agente PLANEJADOR (PlannerAgent)

    - Lê o texto base (submissão) e os pareceres.
    - Sintetiza as principais críticas.
    - Produz um PLANO DE REVISÃO estruturado por seções.
    """

    def build_messages(
        self,
        title: str,
        base_paper: str,
        review: str,
        name: Optional[str] = "",
    ) -> List:
        title = (title or "").strip()
        base_paper = (base_paper or "").strip()
        review = (review or "").strip()
        name = (name or "").strip()

        # Truncamento defensivo
        base_paper = self._truncate_if_needed(base_paper, "MAX_CHARS_BASE")
        review = self._truncate_if_needed(review, "MAX_CHARS_REVIEW")

        id_info = f"Paper ID: {name}\n" if name else ""
        title_info = f"Title: {title}\n" if title else ""

        system_content = (
            "You are an expert scientific writing assistant acting as a PLANNER for article revision.\n"
            "Your job is to read the original draft and the peer reviews, and then produce a clear, "
            "structured REVISION PLAN organized by SECTIONS of the article.\n\n"
            "IMPORTANT PRINCIPLES:\n"
            "1. Work SECTION BY SECTION (e.g., Abstract, Introduction, Related Work, Methods, Experiments, Conclusion).\n"
            "2. For each section that is mentioned in the reviews, summarize the main ISSUES raised by reviewers.\n"
            "3. For each issue, propose concrete REVISION ACTIONS that can be applied directly in the text.\n"
            "4. Preserve the original structure and section ordering of the draft as much as possible.\n"
            "5. Do NOT invent new sections unless the reviewers explicitly request structural changes.\n"
            "6. Focus on incremental edits: rephrase sentences, clarify claims, add missing citations, "
            "and fix inconsistencies, rather than rewriting the entire paper.\n\n"
            "OUTPUT FORMAT (strictly follow this pattern):\n"
            "[SECTION: <section name as in the draft>]\n"
            "- Issues:\n"
            "  - <short bullet describing an issue>\n"
            "  - <another issue>\n"
            "- Revision actions:\n"
            "  - <action that the writer should perform to fix the issue>\n"
            "  - <another action>\n\n"
            "Repeat this block for each relevant section.\n"
        )

        human_content = (
            f"{id_info}{title_info}"
            "ORIGINAL DRAFT (base text):\n"
            "----------------------------------------\n"
            f"{base_paper}\n\n"
            "PEER REVIEWS (comments, critiques, suggestions):\n"
            "----------------------------------------\n"
            f"{review}\n\n"
            "TASK:\n"
            "Based on the draft and the reviews, produce a detailed revision plan organized by sections, "
            "following EXACTLY the output format described in the system message.\n"
            "Do not write the revised article. Only output the structured revision plan.\n"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
        return messages

    def generate_plan(
        self,
        title: str,
        base_paper: str,
        review: str,
        name: Optional[str] = "",
    ) -> Tuple[str, Dict[str, int]]:
        if not base_paper or not review:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        messages = self.build_messages(
            title=title,
            base_paper=base_paper,
            review=review,
            name=name,
        )
        return self.invoke(messages)
