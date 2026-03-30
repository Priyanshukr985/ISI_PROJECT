from langchain_core.prompts import ChatPromptTemplate


class RetrievalGrader:
    """
    Grades retrieved documents for relevance to the user question.
    Returns a plain-text yes/no score.
    """

    def __init__(self, llm_model):
        self.llm = llm_model
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm


    def _build_prompt(self):
        """Create prompt template for grading relevance."""
        system_msg = """
        You are a grader assessing the relevance of a retrieved document to a user question.
        Mark a document as relevant if it directly helps answer the user's question.
        A passing document should contain a definition, explanation, formula, theorem statement,
        or discussion that is actually useful for answering the query.
        The document must match the specific object being asked about.
        For example, a question about a chi-square test is not the same as a chi-square distribution,
        and a question about a distribution is not automatically answered by a test formula.
        If the question asks for a concept such as a theorem, definition, distribution,
        estimator, or statistical idea, then a document that clearly discusses that same concept
        should be marked relevant.
        If the document only mentions the topic loosely or is clearly tangential,
        respond with no.
        If the document contains keywords OR semantic meaning directly useful to the question,
        respond with only one word: yes
        Otherwise respond with only one word: no
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
        Grade a single retrieved document and return plain text yes/no.
        """
        result = self.chain.invoke({"document": document, "question": question})
        text = getattr(result, "content", result).strip().lower()
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        return "no"

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
