

class RetrieverNode:
    """
    Node responsible for retrieving documents based on the question.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state):
        print("---RETRIEVE---")
        question = state['question']
        state.setdefault("rewrite_count", 0)

        docs = self.retriever.invoke(question)
       
        state['documents'] = docs

        return state
    
    
class GraderNode:
    """
    Node that grades relevance of retrieved documents.
    """

    def __init__(self, retrieval_grader):
        self.retrieval_grader = retrieval_grader

    def run(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        question = state['question']
        docs = state['documents']

        filtered = []
        transform_query_required = "No"
        rejected_count = 0

        for doc in docs:
            grade = self.retrieval_grader.grade(doc.page_content, question)
            score = str(grade).strip().lower()

            if score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                rejected_count += 1

        if not filtered:
            transform_query_required = "Yes"
        elif rejected_count and len(filtered) <= 1:
            transform_query_required = "Yes"

        state['documents'] = filtered
        state['transform_query'] = transform_query_required

        return state


class DecisionNode:
    """
    Decides whether to generate or transform query.
    """

    def run(self, state):
        print("---ASSESS GRADED DOCUMENTS---")

        rewrite_count = state.get("rewrite_count", 0)

        if not state.get("documents"):
            if rewrite_count < 1:
                print("---DECISION: TRANSFORM QUERY (NO DOCUMENTS)---")
                return "transform_query"
            print("---DECISION: GENERATE FALLBACK---")
            return "generate"

        if state['transform_query'] == "Yes" and rewrite_count < 1:
            print("---DECISION: TRANSFORM QUERY---")
            return "transform_query"

        print("---DECISION: GENERATE---")
        return "generate"
    


class GeneratorNode:
    """
    Node that generates final answer using RAG chain.
    """

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.unsupported_answer = "I could not find a supported answer in the provided statistics book."

    def _format_source_pages(self, documents):
        pages = sorted(
            {
                doc.metadata.get("page") + 1
                for doc in documents
                if doc.metadata.get("page") is not None
            }
        )
        if not pages:
            return ""
        return "Source pages: " + ", ".join(str(page) for page in pages)

    def _normalize_output(self, output):
        text = str(output).strip()
        if self.unsupported_answer in text:
            return (
                f"{self.unsupported_answer} "
                "Please ask about a topic covered in the book or rephrase the question."
            )
        return text

    def _select_best_documents(self, documents):
        selected = []
        seen_pages = set()
        for doc in documents:
            page = doc.metadata.get("page")
            if page in seen_pages:
                continue
            selected.append(doc)
            if page is not None:
                seen_pages.add(page)
            if len(selected) >= 3:
                break
        return selected or documents[:3]

    def run(self, state):
        print("---GENERATE---")

        question = state['question']
        documents = state['documents']

        if not documents:
            state['generation'] = (
                f"{self.unsupported_answer} "
                "Please ask about a topic covered in the book or rephrase the question."
            )
            return state

        documents = self._select_best_documents(documents)
        context = "\n\n".join(doc.page_content for doc in documents)

        output = self.rag_chain.invoke({"context": context, "question": question})
        output = self._normalize_output(output)
        source_pages = self._format_source_pages(documents)
        if source_pages and self.unsupported_answer not in output:
            output = f"{output}\n\n{source_pages}"
        state['generation'] = output

        return state


class QueryTransformNode:
    """
    Node that rewrites the question for better retrieval.
    """

    def __init__(self, question_rewriter):
        self.question_rewriter = question_rewriter

    def run(self, state):
        print("---TRANSFORM QUERY---")

        new_q = self.question_rewriter.rewrite(state['question'])
        state['question'] = new_q
        state['rewrite_count'] = state.get("rewrite_count", 0) + 1

        return state
