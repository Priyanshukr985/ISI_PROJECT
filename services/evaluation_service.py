from __future__ import annotations

from typing import Dict, List
import logging


class EvaluationService:
    """Runs RAGAS evaluations with safe fallbacks."""

    DEFAULT_SCORES = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }

    def __init__(self, llm=None, embeddings=None):
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        self.embeddings = embeddings
        self._ragas_ready = False
        self._load_dependencies()

    def _load_dependencies(self):
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self.Dataset = Dataset
            self.ragas_evaluate = evaluate
            self.base_metrics = [
                faithfulness,
                answer_relevancy,
            ]
            self.reference_metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            self._ragas_ready = True
        except Exception:
            self.Dataset = None
            self.ragas_evaluate = None
            self.base_metrics = []
            self.reference_metrics = []
            self._ragas_ready = False

    def _default_scores(self) -> Dict[str, float]:
        return dict(self.DEFAULT_SCORES)

    def _normalize_result(self, result) -> Dict[str, float]:
        try:
            if hasattr(result, "to_pandas"):
                row = result.to_pandas().iloc[0].to_dict()
            elif hasattr(result, "to_dict"):
                row = result.to_dict()
            elif isinstance(result, dict):
                row = result
            else:
                row = {}
        except Exception:
            row = {}

        scores = self._default_scores()
        for key in scores:
            value = row.get(key, 0.0)
            try:
                scores[key] = float(value)
            except Exception:
                scores[key] = 0.0
        return scores

    def evaluate_rag_output(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: str | None = None,
    ) -> Dict[str, float]:
        if not self._ragas_ready or not question or not answer:
            return self._default_scores()

        safe_contexts = [context for context in (contexts or []) if isinstance(context, str) and context.strip()]
        if not safe_contexts:
            return self._default_scores()

        try:
            dataset_payload = {
                "question": [question],
                "answer": [answer],
                "contexts": [safe_contexts],
            }
            metrics = self.base_metrics
            if reference_answer and reference_answer.strip():
                dataset_payload["ground_truth"] = [reference_answer.strip()]
                metrics = self.reference_metrics

            dataset = self.Dataset.from_dict(dataset_payload)
            result = self.ragas_evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return self._normalize_result(result)
        except Exception as error:
            self.logger.warning("RAGAS evaluation failed: %s", error)
            return self._default_scores()

    def evaluate_batch(self, samples: List[dict]) -> List[Dict[str, float]]:
        results = []
        for sample in samples or []:
            results.append(
                self.evaluate_rag_output(
                    question=sample.get("question", ""),
                    answer=sample.get("answer", ""),
                    contexts=sample.get("contexts", []),
                )
            )
        return results

    def average_scores(self, evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        if not evaluations:
            return self._default_scores()

        totals = self._default_scores()
        for evaluation in evaluations:
            for key in totals:
                totals[key] += float(evaluation.get(key, 0.0))

        count = len(evaluations)
        return {key: totals[key] / count for key in totals}
