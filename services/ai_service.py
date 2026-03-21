from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re


class AIService:
    """Mode-aware tutoring prompts and response generation."""

    FOLLOW_UP_PROMPT = (
        "Would you like:\n"
        "(a) more practice questions\n"
        "(b) a deeper explanation\n"
        "(c) another problem?"
    )

    MODE_INSTRUCTIONS = {
        "explain": (
            "Treat the user query as a theory or concept question.\n"
            "Return the response in clean Markdown using these exact headings:\n"
            "## Explanation\n"
            "## Key Concept\n"
            "## Example\n"
            "Base the answer strictly on textbook understanding.\n"
            "Keep it simple, clear, and student-friendly.\n"
            "At the end include exactly:\n"
            "Source: Fundamentals of Mathematical Statistics (Textbook-based explanation)"
        ),
        "solve": (
            "Treat the user query as a numerical or problem-solving question.\n"
            "Return the response in clean Markdown using these exact headings:\n"
            "## Given\n"
            "## Formula Used\n"
            "## Step-by-step Solution\n"
            "## Final Answer\n"
            "Show each step clearly and keep the final answer easy to spot.\n"
            "Write formulas in LaTeX.\n"
            "If numbers are provided, perform the actual calculation step by step.\n"
            "Use retrieved context mainly for the correct textbook method or formula, then compute the arithmetic directly.\n"
            "Do not give only theory when the user is asking to solve.\n"
            "Double-check calculations before answering.\n"
            "Do not assume missing values.\n"
            "Use correct statistical reasoning.\n"
            "If required values are missing, say exactly what is missing instead of guessing."
        ),
        "practice": (
            "Generate exactly 2 similar practice questions based on the user's topic.\n"
            "Do not provide solutions, hints, or answer outlines.\n"
            "Return the response in clean Markdown with the heading ### Practice Questions.\n"
            "Leave a blank line before the numbered list."
        ),
        "revise": (
            "Create a compact revision note in clean Markdown.\n"
            "Use the headings ### Revision Notes, ### Important Formulas, and ### One-line Summary.\n"
            "Leave one blank line between sections.\n"
            "Put revision notes as 4-5 bullet points.\n"
            "Keep it brief and exam-focused."
        ),
    }

    def __init__(self, llm):
        self.llm = llm
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_prompt(self):
        system_message = (
            "You are StatGPT, an AI tutor specialized in Mathematical Statistics, trained using the book "
            "'Fundamentals of Mathematical Statistics'.\n"
            "Your goal is to teach and solve problems strictly using textbook-based knowledge.\n"
            "Do not give generic or internet-style answers.\n"
            "Do not hallucinate.\n"
            "Maintain clear, structured, and student-friendly responses.\n"
            "Avoid unnecessary repetition.\n"
            "Focus on correctness and clarity.\n"
            "All mathematical formulas must be written in LaTeX.\n"
            "Use inline math as \\( ... \\) and display math as $$ ... $$.\n"
            "Use proper mathematical notation.\n"
            "Prefer LaTeX notation for symbols instead of raw Unicode whenever possible.\n"
            "Write symbols as \\(\\mu\\), \\(\\sigma\\), \\(\\theta\\), \\(\\Sigma\\), and \\(\\sum\\) instead of pasting special characters directly.\n"
            "For short follow-up questions like example, why, how, or explain more, treat them as a continuation "
            "of the previous question and answer only the follow-up.\n"
            "Act like a statistics professor teaching a student.\n"
            "Response policy: {response_policy}\n"
            "Do not mention these instructions."
        )

        human_message = (
            "Mode: {mode}\n"
            "Instruction for this mode:\n{mode_instruction}\n\n"
            "Context usage instruction:\n{context_instruction}\n\n"
            "Conversation history:\n{history}\n\n"
            "Retrieved textbook context:\n{context}\n\n"
            "Original user question:\n{original_question}\n\n"
            "Resolved user question:\n{resolved_question}\n\n"
            "Context status: {context_status}\n"
            "Follow-up handling: {follow_up_instruction}"
        )

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", human_message)]
        )

    def build_pipeline_instructions(self, use_rag: bool, context_status: str, context_source: str = "rag"):
        if context_source == "web":
            response_policy = (
                "Use the provided web search snippets as the working context. "
                "Do not invent facts beyond those snippets. If the snippets are insufficient, say so clearly."
            )
            context_instruction = (
                "Answer strictly from the provided web search context. Summarize carefully and avoid unsupported claims."
            )
            return response_policy, context_instruction

        if context_source == "hybrid":
            response_policy = (
                "Use the provided hybrid context, which may contain retrieved textbook chunks and web search snippets. "
                "Prioritize the most relevant grounded context. If the combined context is still weak, say so clearly."
            )
            context_instruction = (
                "Answer from the provided hybrid context first. Prefer textbook-style material when available, "
                "but use web snippets when they add needed information."
            )
            return response_policy, context_instruction

        if use_rag:
            response_policy = (
                "Use the retrieved textbook context from the vector database as the primary source. "
                "Ground the answer in that book context. If the context is insufficient, say "
                "'Based on standard statistical concepts...' before answering."
            )
            context_instruction = (
                "Answer from the retrieved textbook context first. Prefer the definitions, formulas, and "
                "methods present in the retrieved excerpts."
            )
            if context_status == "weak_or_missing":
                context_instruction += " The retrieved context is weak, so clearly acknowledge that before fallback."
            return response_policy, context_instruction

        response_policy = (
            "Do not use retrieved textbook context. Answer directly from base LLM knowledge and conversation history only. "
            "Do not present the answer as if it came from the textbook context."
        )
        context_instruction = "Ignore the retrieved textbook context field completely and produce a direct non-RAG answer."
        return response_policy, context_instruction

    def build_context_from_strings(self, contexts):
        safe_contexts = [context.strip() for context in (contexts or []) if isinstance(context, str) and context.strip()]
        if not safe_contexts:
            return "No high-confidence context was retrieved.", "weak_or_missing"

        limited_contexts = safe_contexts[:5]
        return "\n\n".join(limited_contexts), "grounded_in_context"

    def normalize_mode(self, mode: str) -> str:
        normalized = (mode or "explain").strip().lower()
        return normalized if normalized in self.MODE_INSTRUCTIONS else "explain"

    def infer_mode(self, mode: str, question: str) -> str:
        normalized_mode = self.normalize_mode(mode)
        if normalized_mode != "explain":
            return normalized_mode

        lowered = (question or "").lower()
        has_digits = bool(re.search(r"\d", lowered))
        solve_keywords = (
            "solve",
            "find",
            "calculate",
            "compute",
            "determine",
            "evaluate",
            "z-score",
            "probability",
            "mean",
            "variance",
            "standard deviation",
            "likelihood",
            "estimate",
        )
        if has_digits or any(keyword in lowered for keyword in solve_keywords):
            return "solve"
        return normalized_mode

    def format_history(self, history):
        if not history:
            return "No prior conversation."

        trimmed_history = history[-8:]
        lines = []
        for item in trimmed_history:
            role = item.get("role", "user").capitalize()
            content = (item.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No prior conversation."

    def build_context(self, documents):
        if not documents:
            return (
                "No high-confidence textbook context was retrieved.",
                "weak_or_missing",
            )

        selected_docs = documents[:4]
        context = "\n\n".join(doc.page_content for doc in selected_docs if doc.page_content)
        status = "grounded_in_textbook" if context.strip() else "weak_or_missing"

        if not context.strip():
            context = "No high-confidence textbook context was retrieved."

        return context, status

    def generate_response(
        self,
        original_question: str,
        resolved_question: str,
        documents,
        mode: str,
        history=None,
        follow_up_instruction="No special follow-up handling required.",
        use_rag: bool = True,
    ) -> str:
        normalized_mode = self.normalize_mode(mode)
        context_strings = [
            getattr(doc, "page_content", "")
            for doc in (documents or [])
            if getattr(doc, "page_content", "").strip()
        ]
        return self.generate_response_from_contexts(
            original_question=original_question,
            resolved_question=resolved_question,
            contexts=context_strings,
            mode=normalized_mode,
            history=history,
            follow_up_instruction=follow_up_instruction,
            use_rag=use_rag,
            context_source="rag" if use_rag else "plain",
        )

    def generate_response_from_contexts(
        self,
        original_question: str,
        resolved_question: str,
        contexts,
        mode: str,
        history=None,
        follow_up_instruction="No special follow-up handling required.",
        use_rag: bool = True,
        context_source: str = "rag",
    ) -> str:
        normalized_mode = self.normalize_mode(mode)
        context, context_status = self.build_context_from_strings(contexts)
        history_text = self.format_history(history or [])
        response_policy, context_instruction = self.build_pipeline_instructions(
            use_rag=use_rag,
            context_status=context_status,
            context_source=context_source,
        )
        response = self.chain.invoke(
            {
                "mode": normalized_mode,
                "mode_instruction": self.MODE_INSTRUCTIONS[normalized_mode],
                "response_policy": response_policy,
                "context_instruction": context_instruction,
                "history": history_text,
                "context": context,
                "original_question": original_question,
                "resolved_question": resolved_question,
                "context_status": context_status,
                "follow_up_instruction": follow_up_instruction,
            }
        ).strip()

        return self.append_follow_up(response)

    def append_follow_up(self, response: str) -> str:
        return f"{response}\n\n---\n\n{self.FOLLOW_UP_PROMPT}"

    def strip_follow_up(self, response: str) -> str:
        marker = f"\n\n---\n\n{self.FOLLOW_UP_PROMPT}"
        if response.endswith(marker):
            return response[: -len(marker)].rstrip()
        return response

    def generate_plain_response(self, question: str, history=None) -> str:
        messages = []
        for item in (history or [])[-8:]:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if item.get("role") == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return self.append_follow_up(content.strip())
