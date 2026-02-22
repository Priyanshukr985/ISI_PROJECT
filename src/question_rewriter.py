from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QuestionRewriter:
    """
    Rewrites user questions so they are more semantically clear
    and optimized for information retrieval or web search.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ------------------------------------------
    # PRIVATE METHOD: Build Prompt
    # ------------------------------------------
    def _build_prompt(self):
        system_msg = """
        You are a question re-writer that converts an input question
        into a better version optimized for semantic retrieval and web search.
        Understand the meaning and rewrite the question with improved clarity.
        """

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "human",
                    "Original Question:\n{question}\n\nRewrite the question clearly and precisely:",
                ),
            ]
        )

    # ------------------------------------------
    # PUBLIC METHOD: Rewrite Question
    # ------------------------------------------
    def rewrite(self, question: str) -> str:
        """Rewrite the question using LLM."""
        return self.chain.invoke({"question": question})