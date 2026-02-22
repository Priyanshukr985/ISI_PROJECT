

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

        for doc in docs:
            grade = self.retrieval_grader.grade(doc.page_content, question)
            # grade = score.binary_score

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
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

        if state['transform_query'] == "Yes":
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

    def run(self, state):
        print("---GENERATE---")

        question = state['question']
        documents = state['documents']

        # Format documents internally
        context = "\n\n".join(doc.page_content for doc in documents)

        output = self.rag_chain.invoke({"context": context, "question": question})
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

        return state