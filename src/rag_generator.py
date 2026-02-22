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
        system_message = """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, say 'I don't know"""

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
        return "\n\n".join(doc.page_content for doc in docs)

    def generate(self, docs, question):
        """
        Run RAG answering:
        - Format docs
        - Feed context + question to LLM
        - Return generated answer
        """
        context = self.format_docs(docs)
        return self.chain.invoke({"context": context, "question": question})