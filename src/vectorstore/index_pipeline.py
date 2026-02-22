from src.vectorstore.vector import PDFLoader , TextChunker , HFEmbedding , FAISSStore

class IndexBuilder:
    def __init__(self, pdf_path, chunk_size=1000 , chunk_overlap = 200, device = "cpu"):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

    def build_index(self):
        # 1. Load PDF
        print("[1] Loading PDF...")
        docs = PDFLoader(self.pdf_path).load()

        # 2. Chunk
        print("[2] Splitting PDF into chunks... ")
        chunks = TextChunker(self.chunk_size, self.chunk_overlap).split(docs)

        # 3. Embeddings
        print("[3] Loading embedding model... ")
        embedding_model = HFEmbedding(device=self.device).load()

        # 4. Build FAISS store
        print("[4] Building FAISS vectorstore...")
        faiss_store = FAISSStore(embedding_model, index_path="faiss_index")
        faiss_store.build(chunks)

        # SAVE VECTOR DB LOCALLY
        print("[5] Saving vectorstore...")
        faiss_store.save()






