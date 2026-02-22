from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGrader:
    """
    Grades retrieved documents for relevance to the user question.
    Uses Groq LLM with structured output.
    """

    def __init__(self, llm_model):
        self.llm = llm_model
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm


    def _build_prompt(self):
        """Create prompt template for grading relevance."""
        system_msg = """
        You are a grader assessing the relevance of a retrieved document to a user question.
        If the document contains keywords OR semantic meaning related to the question,
        respond 'yes'. Otherwise respond 'no'.
        """

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "Retrieved document:\n\n{document}\n\nUser question:\n{question}")
            ]
        )

    # ------------------------------
    # PUBLIC METHODS
    # ------------------------------

    def grade(self, document: str, question: str):
        """
        Grade a single retrieved document.
        Returns GradeDocuments(binary_score='yes'/'no')
        """
        return self.chain.invoke({"document": document, "question": question})

    def grade_all(self, documents, question: str):
        """
        Grade a list of LangChain Document objects.
        Returns list of (doc, grade)
        """
        results = []
        for doc in documents:
            doc_text = doc.page_content
            score = self.grade(doc_text, question)
            results.append((doc, score))
        return results