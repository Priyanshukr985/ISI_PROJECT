from langchain_community.retrievers import BM25Retriever
import logging

from src.graph_builder import WorkflowBuilder
from src.graph_node import (
    DecisionNode,
    GeneratorNode,
    GraderNode,
    QueryTransformNode,
    RetrieverNode,
)
from src.question_rewriter import QuestionRewriter
from src.retrieval_grader import RetrievalGrader


class RagService:
    """Coordinates textbook retrieval workflow and final tutor responses."""

    PIPELINE_MODES = {"with_rag", "without_rag", "compare"}

    def __init__(
        self,
        retriever,
        vectordb,
        llm,
        ai_service,
        memory_service,
        solver_service=None,
        router_service=None,
        web_search_service=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.retriever = retriever
        self.vectordb = vectordb
        self.llm = llm
        self.ai_service = ai_service
        self.memory_service = memory_service
        self.solver_service = solver_service
        self.router_service = router_service
        self.web_search_service = web_search_service
        self.top_k = 5

        document_grader = RetrievalGrader(llm)
        question_rewriter = QuestionRewriter(llm=llm)
        self.corpus_documents = self._load_corpus_documents()
        self.bm25_retriever = self._build_bm25_retriever()

        builder = WorkflowBuilder(
            retrieve_node=RetrieverNode(self.retriever),
            grade_node=GraderNode(document_grader),
            generate_node=GeneratorNode(self.ai_service),
            transform_node=QueryTransformNode(question_rewriter),
            decision_node=DecisionNode(),
        )
        self.app_graph = builder.build()

    def _load_corpus_documents(self):
        try:
            docstore = getattr(self.vectordb, "docstore", None)
            store_dict = getattr(docstore, "_dict", {}) if docstore else {}
            return list(store_dict.values())
        except Exception as error:
            self.logger.warning("Could not load corpus documents: %s", error)
            return []

    def _build_bm25_retriever(self):
        if not self.corpus_documents:
            return None
        try:
            retriever = BM25Retriever.from_documents(self.corpus_documents)
            retriever.k = self.top_k
            return retriever
        except Exception as error:
            self.logger.warning("Could not build BM25 retriever: %s", error)
            return None

    def _resolve_modes(self, mode: str, tutor_mode: str, question: str):
        normalized_mode = (mode or "").strip().lower()
        if normalized_mode in self.PIPELINE_MODES:
            pipeline_mode = normalized_mode
            selected_tutor_mode = tutor_mode or "explain"
        else:
            pipeline_mode = "with_rag"
            selected_tutor_mode = mode or tutor_mode or "explain"

        effective_tutor_mode = self.ai_service.infer_mode(selected_tutor_mode, question)
        legacy_mode = normalized_mode not in self.PIPELINE_MODES
        return pipeline_mode, effective_tutor_mode, legacy_mode

    def _answer_with_rag(self, original_question, resolved_question, tutor_mode, history, follow_up_instruction, allow_solver):
        if allow_solver and tutor_mode == "solve" and self.solver_service is not None:
            solved_reply = self.solver_service.solve(
                resolved_question,
                self.ai_service.FOLLOW_UP_PROMPT,
            )
            if solved_reply:
                return solved_reply

        state = {
            "question": resolved_question,
            "original_question": original_question,
            "resolved_question": resolved_question,
            "mode": tutor_mode,
            "history": history,
            "follow_up_instruction": follow_up_instruction,
            "is_follow_up": bool(follow_up_instruction),
        }
        result = self.app_graph.invoke(state)
        return result["generation"]

    def _answer_without_rag(self, original_question, resolved_question, tutor_mode, history, follow_up_instruction):
        return self.ai_service.generate_plain_response(
            question=resolved_question,
            history=history,
        )

    def _clean_contexts(self, documents):
        contexts = []
        seen = set()
        for doc in documents or []:
            content = doc if isinstance(doc, str) else getattr(doc, "page_content", "")
            cleaned = " ".join((content or "").split()).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            contexts.append(cleaned)
        return contexts[: self.top_k]

    def hybrid_retrieval(self, query):
        vector_docs = []
        bm25_docs = []

        try:
            vector_docs = self.retriever.invoke(query) or []
        except Exception as error:
            self.logger.warning("Vector retrieval failed: %s", error)
            vector_docs = []

        if self.bm25_retriever is not None:
            try:
                bm25_docs = self.bm25_retriever.invoke(query) or []
            except Exception as error:
                self.logger.warning("BM25 retrieval failed: %s", error)
                bm25_docs = []

        return self._clean_contexts([*vector_docs, *bm25_docs])

    def _decide_route(self, query: str) -> str:
        if self.router_service is None:
            return "rag"
        try:
            route = self.router_service.decide_route(query)
            if route in {"rag", "web", "hybrid"}:
                return route
        except Exception as error:
            self.logger.warning("Route decision failed: %s", error)
        return "rag"

    def _search_web(self, query: str):
        if self.web_search_service is None:
            return []
        try:
            return self.web_search_service.search_web(query)
        except Exception as error:
            self.logger.warning("Web search failed: %s", error)
            return []

    def _generate_contextual_answer(
        self,
        original_question,
        resolved_question,
        tutor_mode,
        history,
        follow_up_instruction,
        contexts,
        route,
    ):
        return self.ai_service.generate_response_from_contexts(
            original_question=original_question,
            resolved_question=resolved_question,
            contexts=contexts,
            mode=tutor_mode,
            history=history,
            follow_up_instruction=follow_up_instruction,
            use_rag=True,
            context_source=route,
        )

    def _prepare_query(self, question: str, mode: str = "explain", session_id=None, tutor_mode=None):
        session_key = self.memory_service.ensure_session(session_id)
        history = self.memory_service.get_history(session_key)
        query_plan = self.memory_service.resolve_user_query(question, history)
        pipeline_mode, effective_tutor_mode, legacy_mode = self._resolve_modes(
            mode,
            tutor_mode,
            query_plan["resolved_question"],
        )
        return {
            "session_key": session_key,
            "history": history,
            "query_plan": query_plan,
            "pipeline_mode": pipeline_mode,
            "tutor_mode": effective_tutor_mode,
            "legacy_mode": legacy_mode,
        }

    def generate_rag_response(self, query: str, session_id=None, tutor_mode="explain") -> dict:
        return self.generate_smart_response(
            query=query,
            session_id=session_id,
            tutor_mode=tutor_mode,
        )

    def generate_smart_response(self, query: str, session_id=None, tutor_mode="explain") -> dict:
        prepared = self._prepare_query(
            question=query,
            mode="with_rag",
            session_id=session_id,
            tutor_mode=tutor_mode,
        )
        resolved_question = prepared["query_plan"]["resolved_question"]
        route = self._decide_route(resolved_question)
        rag_contexts = self.hybrid_retrieval(resolved_question)
        web_contexts = self._search_web(resolved_question)

        if route == "web":
            contexts = web_contexts or rag_contexts
            route = "web" if web_contexts else "rag"
        elif route == "hybrid":
            contexts = self._clean_contexts([*rag_contexts, *web_contexts])
        else:
            contexts = rag_contexts

        if not contexts:
            contexts = ["No high-confidence textbook context was retrieved."]
            route = "rag"

        if prepared["tutor_mode"] == "solve" and self.solver_service is not None:
            solved_reply = self.solver_service.solve(
                resolved_question,
                self.ai_service.FOLLOW_UP_PROMPT,
            )
            if solved_reply:
                answer = solved_reply
            else:
                answer = self._generate_contextual_answer(
                    original_question=query,
                    resolved_question=resolved_question,
                    tutor_mode=prepared["tutor_mode"],
                    history=prepared["history"],
                    follow_up_instruction=prepared["query_plan"]["follow_up_instruction"],
                    contexts=contexts,
                    route=route,
                )
        else:
            answer = self._generate_contextual_answer(
                original_question=query,
                resolved_question=resolved_question,
                tutor_mode=prepared["tutor_mode"],
                history=prepared["history"],
                follow_up_instruction=prepared["query_plan"]["follow_up_instruction"],
                contexts=contexts,
                route=route,
            )

        return {
            "answer": answer,
            "contexts": contexts,
            "route": route,
            "session_id": prepared["session_key"],
            "resolved_question": resolved_question,
            "history": prepared["history"],
            "follow_up_instruction": prepared["query_plan"]["follow_up_instruction"],
            "tutor_mode": prepared["tutor_mode"],
        }

    def answer_question(self, question: str, mode: str = "explain", session_id=None, tutor_mode=None) -> dict:
        prepared = self._prepare_query(
            question=question,
            mode=mode,
            session_id=session_id,
            tutor_mode=tutor_mode,
        )
        session_key = prepared["session_key"]
        history = prepared["history"]
        query_plan = prepared["query_plan"]
        pipeline_mode = prepared["pipeline_mode"]
        effective_tutor_mode = prepared["tutor_mode"]
        legacy_mode = prepared["legacy_mode"]

        common_args = {
            "original_question": question,
            "resolved_question": query_plan["resolved_question"],
            "tutor_mode": effective_tutor_mode,
            "history": history,
            "follow_up_instruction": query_plan["follow_up_instruction"],
        }

        if pipeline_mode == "without_rag":
            reply = self._answer_without_rag(**common_args)
            self.memory_service.add_turn(session_key, question, reply)
            return {
                "reply": reply,
                "session_id": session_key,
                "mode": pipeline_mode,
                "tutor_mode": effective_tutor_mode,
            }

        if pipeline_mode == "compare":
            rag_result = self.generate_smart_response(
                query=question,
                session_id=session_key,
                tutor_mode=effective_tutor_mode,
            )
            with_rag = rag_result["answer"]
            without_rag = self._answer_without_rag(**common_args)
            with_rag_clean = self.ai_service.strip_follow_up(with_rag)
            without_rag_clean = self.ai_service.strip_follow_up(without_rag)
            combined_reply = (
                f"## With RAG\n\n{with_rag_clean}\n\n"
                f"## Without RAG\n\n{without_rag_clean}"
            )
            self.memory_service.add_turn(session_key, question, combined_reply)
            return {
                "question": question,
                "with_rag": with_rag_clean,
                "without_rag": without_rag_clean,
                "contexts": rag_result["contexts"],
                "route": rag_result["route"],
                "session_id": session_key,
                "mode": pipeline_mode,
                "tutor_mode": effective_tutor_mode,
            }

        if pipeline_mode == "with_rag":
            rag_result = self.generate_smart_response(
                query=question,
                session_id=session_key,
                tutor_mode=effective_tutor_mode,
            )
            reply = rag_result["answer"]
            self.memory_service.add_turn(session_key, question, reply)
            return {
                "reply": reply,
                "session_id": session_key,
                "mode": pipeline_mode,
                "tutor_mode": effective_tutor_mode,
            }

        allow_solver = legacy_mode
        reply = self._answer_with_rag(**common_args, allow_solver=allow_solver)
        self.memory_service.add_turn(session_key, question, reply)
        return {
            "reply": reply,
            "session_id": session_key,
            "mode": pipeline_mode if not legacy_mode else effective_tutor_mode,
            "tutor_mode": effective_tutor_mode,
        }
