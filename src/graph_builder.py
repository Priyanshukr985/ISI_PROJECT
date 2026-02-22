from langgraph.graph import END, START, StateGraph

from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    transform_query: str
    generation: str
    documents: List[str]
    

class WorkflowBuilder:
    """
    Modular LangGraph workflow builder.
    Takes modular nodes and constructs the RAG workflow graph.
    """

    def __init__(self, retrieve_node, grade_node, generate_node,transform_node, decision_node):
        self.retrieve_node = retrieve_node
        self.grade_node = grade_node
        self.generate_node = generate_node
        self.transform_node = transform_node
        self.decision_node = decision_node

    def build(self):
        workflow = StateGraph(GraphState)

        # Register nodes
        workflow.add_node("retrieve", self.retrieve_node.run)
        workflow.add_node("grade_documents", self.grade_node.run)
        workflow.add_node("generate", self.generate_node.run)
        workflow.add_node("transform_query", self.transform_node.run)

        # Build edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional
        workflow.add_conditional_edges(
            "grade_documents",
            self.decision_node.run,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        workflow.add_edge("transform_query", "grade_documents")
        workflow.add_edge("generate", END)

        return workflow.compile()