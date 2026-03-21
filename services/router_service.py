class RouterService:
    """Rule-based router for deciding retrieval strategy."""

    WEB_KEYWORDS = ("who is", "latest", "news", "today", "recent", "current")
    RAG_KEYWORDS = ("define", "explain", "theorem", "lemma", "corollary", "distribution")

    def decide_route(self, query: str) -> str:
        lowered = (query or "").strip().lower()
        if any(keyword in lowered for keyword in self.WEB_KEYWORDS):
            return "web"
        if any(keyword in lowered for keyword in self.RAG_KEYWORDS):
            return "rag"
        return "hybrid"
