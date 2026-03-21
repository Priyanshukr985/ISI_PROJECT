from datetime import datetime, timezone
import logging
import os
from time import perf_counter

from services.ai_service import AIService
from services.evaluation_service import EvaluationService
from services.image_service import ImageService
from services.logging_service import LoggingService
from services.memory_service import MemoryService
from services.rag_service import RagService
from services.router_service import RouterService
from services.solver_service import SolverService
from services.web_search_service import WebSearchService
from src.vectorstore.vector import HFEmbedding, FAISSStore
from src.llm_model import LLM_Loader
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

required_env = ("GROQ_API_KEY", "LLM_MODEL")
missing_env = [key for key in required_env if not os.getenv(key)]
if missing_env:
    raise RuntimeError(
        "Missing required environment variables: " + ", ".join(missing_env)
    )

llm_loader = LLM_Loader()
llm = llm_loader.load()

embedding_loader = HFEmbedding()
embedding_model = embedding_loader.load()

faiss_store = FAISSStore(embedding_model, index_path="faiss_index")
vectordb = faiss_store.load()

retriever = faiss_store.get_retriever(vectordb, search_kwargs={"k": 4})
ai_service = AIService(llm)
memory_service = MemoryService(max_messages=8)
image_service = ImageService()
solver_service = SolverService()
evaluation_service = EvaluationService(llm=llm, embeddings=embedding_model)
logging_service = LoggingService(log_path="logs/rag_logs.json")
router_service = RouterService()
web_search_service = WebSearchService()
rag_service = RagService(
    retriever=retriever,
    vectordb=vectordb,
    llm=llm,
    ai_service=ai_service,
    memory_service=memory_service,
    solver_service=solver_service,
    router_service=router_service,
    web_search_service=web_search_service,
)


def _build_compare_response(result):
    started_at = perf_counter()
    evaluation = evaluation_service.evaluate_rag_output(
        question=result["question"],
        answer=result["with_rag"],
        contexts=result.get("contexts", []),
    )
    latency_ms = round((perf_counter() - started_at) * 1000, 2)

    try:
        logging_service.log_interaction(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": result["question"],
                "rag_answer": result["with_rag"],
                "non_rag_answer": result["without_rag"],
                "route": result.get("route", "rag"),
                "contexts": result.get("contexts", []),
                "evaluation": evaluation,
                "latency_ms": latency_ms,
            }
        )
    except Exception as error:
        logger.warning("Failed to write compare log: %s", error)

    return {
        "question": result["question"],
        "with_rag": result["with_rag"],
        "without_rag": result["without_rag"],
        "evaluation": evaluation,
        "mode": result["mode"],
        "tutor_mode": result["tutor_mode"],
        "session_id": result["session_id"],
    }


@app.route("/")
def home():
    """Serve your frontend HTML if using Flask templates."""
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives JSON: { "message": "text from user", "mode": "with_rag|without_rag|compare|legacy tutor mode", "tutor_mode": "explain|solve|practice|revise", "session_id": "id" }
    Returns: { "reply": "Tutor response", "mode": "selected mode", "session_id": "id" }
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "")
        mode = data.get("mode", "explain")
        tutor_mode = data.get("tutor_mode")
        session_id = data.get("session_id")

        if not user_message.strip():
            return jsonify({"reply": "Please type a message."})

        result = rag_service.answer_question(
            user_message,
            mode=mode,
            session_id=session_id,
            tutor_mode=tutor_mode,
        )

        if result["mode"] == "compare":
            return jsonify(_build_compare_response(result))

        return jsonify(
            {
                "reply": result["reply"],
                "mode": result["mode"],
                "tutor_mode": result["tutor_mode"],
                "session_id": result["session_id"],
            }
        )

    except Exception as error:
        logger.exception("Chat endpoint failed: %s", error)
        return jsonify({"reply": "Server error occurred. Try again later."})


@app.route("/chat/image", methods=["POST"])
def chat_image():
    """Accepts an image question, extracts text, and routes it through the tutor pipeline."""
    try:
        uploaded_image = request.files.get("image")
        mode = request.form.get("mode", "with_rag")
        tutor_mode = request.form.get("tutor_mode", "solve")
        session_id = request.form.get("session_id")
        extra_prompt = (request.form.get("message") or "").strip()

        if uploaded_image is None or not uploaded_image.filename:
            return jsonify({"reply": "Please upload a question image."}), 400

        extracted_text = image_service.extract_text(uploaded_image)
        user_message = extracted_text if not extra_prompt else f"{extra_prompt}\n\nExtracted question: {extracted_text}"
        result = rag_service.answer_question(
            user_message,
            mode=mode,
            session_id=session_id,
            tutor_mode=tutor_mode,
        )

        if result["mode"] == "compare":
            compare_payload = _build_compare_response(result)
            compare_payload["extracted_text"] = extracted_text
            return jsonify(compare_payload)

        return jsonify(
            {
                "reply": result["reply"],
                "mode": result["mode"],
                "tutor_mode": result["tutor_mode"],
                "session_id": result["session_id"],
                "extracted_text": extracted_text,
            }
        )
    except ValueError as e:
        return jsonify({"reply": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"reply": str(e)}), 501
    except Exception as error:
        logger.exception("Image endpoint failed: %s", error)
        return jsonify({"reply": "Image processing failed. Try again later."}), 500


#  RUN SERVER 
if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    app.run(debug=debug_enabled)
