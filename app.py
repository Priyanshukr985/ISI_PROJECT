from src.vectorstore.index_pipeline import IndexBuilder
from src.vectorstore.vector import PDFLoader , TextChunker , HFEmbedding , FAISSStore
from src.llm_model import LLM_Loader
from src.retrieval_grader import RetrievalGrader
from src.rag_generator import Rag_Generator
from src.question_rewriter import QuestionRewriter
from src.graph_node import RetrieverNode, GraderNode, GeneratorNode, QueryTransformNode, DecisionNode
from src.graph_builder import WorkflowBuilder
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# IndexBuilder("Stat_Book.pdf").build_index()

llm_loader = LLM_Loader()
llm = llm_loader.load()

embedding_loader = HFEmbedding()
embedding_model = embedding_loader.load()

faiss_store = FAISSStore(embedding_model, index_path="faiss_index")
vectordb = faiss_store.load()

retriever = faiss_store.get_retriever(vectordb)

# Instantiate functional modules
document_grader = RetrievalGrader(llm)
question_rewriter = QuestionRewriter(llm=llm)
rag_generator = Rag_Generator(llm)

# Create Node Objects
retrieve_node = RetrieverNode(retriever)
grade_node = GraderNode(document_grader)
generate_node = GeneratorNode(rag_generator.chain)
transform_node = QueryTransformNode(question_rewriter)
decision_node = DecisionNode()

# Build LangGraph Workflow
Builder = WorkflowBuilder(retrieve_node=retrieve_node,
                          grade_node=grade_node,
                          generate_node=generate_node,
                          transform_node=transform_node,
                          decision_node=decision_node)

app_graph = Builder.build()


# Run Graph
# state = {"question": "Central limit theorem"}
# result = app_graph.invoke(state)

# print("\n=== FINAL RESULT ===")
# print(result["generation"])


@app.route("/")
def home():
    """Serve your frontend HTML if using Flask templates."""
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives JSON: { "message": "text from user" }
    Returns: { "reply": "AI doctor response" }
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"reply": "Please type a message."})

        # LLM RESPONSE
        state = {"question": user_message}
        ai_reply = app_graph.invoke(state)['generation']

        return jsonify({"reply": ai_reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "Server error occurred. Try again later."})


#  RUN SERVER 
if __name__ == "__main__":
    app.run(debug=True)