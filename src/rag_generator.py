from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Rag_Generator:
    """
    Generates final answers using retrieved context + user query.
    Uses an LLM (Groq/OpenAI/HF) passed via dependency injection.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    # -------------------------------------
    # PRIVATE METHODS
    # -------------------------------------
    def _build_prompt(self):
        """Return structured ChatPromptTemplate."""
        system_message = """You are a statistics expert.

Your task is to generate clear, concise, and exam-ready answers.

Follow these STRICT rules:
1. Start with a precise definition (maximum 2-3 lines).
2. Provide the standard mathematical form (if applicable).
3. Add 1-2 key points such as interpretation, properties, or uses.
4. Keep the answer structured and easy to read.
5. Use simple and clear language.
6. Use bullet points for key points.
7. Avoid repetition completely.
8. Do NOT include advanced extensions, derivations, or unrelated theory unless explicitly asked.
9. If context is provided (from RAG), use only relevant information and rewrite it cleanly -- do NOT copy text directly.

IMPORTANT KNOWLEDGE RULES:
10. If standard definitions or formulas are missing from the context, use your own knowledge to complete them.
11. Do NOT say "Not applicable" for well-known mathematical forms (e.g., CLT, chi-square, normal distribution).
12. Only say "Not applicable" if the concept truly has no mathematical expression.
13. Prefer correct standard statistical definitions over incomplete context.

CONSISTENCY RULES:
14. Ensure all statements are mathematically correct.
15. Avoid vague or misleading statements (e.g., incorrect degrees of freedom).
16. Keep answers short, exam-focused, and precise.
17. Write mathematical notation in clean LaTeX whenever possible.
18. For formulas, use standard MathJax-friendly delimiters such as $$...$$ for display equations and $...$ for inline symbols.

Output format:

Definition: <clear and concise definition>

Mathematical Form:
<formula (or "Not applicable" only if truly none exists)>

Key Points:
• point 1
• point 2"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Retrieved document:\n\n{context}\n\nUser question:\n{question}")
            ]
        )

    # -------------------------------------
    # PUBLIC METHODS
    # -------------------------------------
    def format_docs(self, docs):
        """
        Convert list of LangChain Document objects into a single text block.
        """
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            if page is not None:
                formatted.append(f"[Page {page + 1}]\n{doc.page_content}")
            else:
                formatted.append(doc.page_content)
        return "\n\n".join(formatted)

    def generate(self, docs, question):
        """
        Run RAG answering:
        - Format docs
        - Feed context + question to LLM
        - Return generated answer
        """
        context = self.format_docs(docs)
        return self.chain.invoke({"context": context, "question": question})
