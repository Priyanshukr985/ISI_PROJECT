

class RetrieverNode:
    """
    Node responsible for retrieving documents based on the question.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state):
        print("---RETRIEVE---")
        question = state['question']
        docs = self.retriever.invoke(question)
        state['documents'] = docs
        state.setdefault('rewrite_attempts', 0)

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
        rewrite_attempts = state.get('rewrite_attempts', 0)

        for doc in docs:
            grade = self.retrieval_grader.grade(doc.page_content, question)

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        state['documents'] = filtered
        state['transform_query'] = "Yes" if not filtered and rewrite_attempts < 1 else "No"

        return state


class DecisionNode:
    """
    Decides whether to generate or transform query.
    """

    def run(self, state):
        print("---ASSESS GRADED DOCUMENTS---")

        if state['transform_query'] == "Yes":
            print("---DECISION: TRANSFORM QUERY---")
            return "transform_query"

        print("---DECISION: GENERATE---")
        return "generate"
    


class GeneratorNode:
    """
    Node that generates final answer using RAG chain.
    """

    def __init__(self, ai_service):
        self.ai_service = ai_service

    def run(self, state):
        print("---GENERATE---")

        original_question = state.get('original_question', state['question'])
        resolved_question = state.get('resolved_question', state['question'])
        documents = state['documents']
        mode = state.get('mode', 'explain')
        history = state.get('history', [])
        follow_up_instruction = state.get(
            'follow_up_instruction',
            'No special follow-up handling required.',
        )
        output = self.ai_service.generate_response(
            original_question=original_question,
            resolved_question=resolved_question,
            documents=documents,
            mode=mode,
            history=history,
            follow_up_instruction=follow_up_instruction,
        )
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
        state['rewrite_attempts'] = state.get('rewrite_attempts', 0) + 1

        return state
