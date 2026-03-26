import chromadb
import hashlib


class VectorStore:
    def __init__(self, persist_dir="chroma_db", collection_name="rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def generate_id(self, text, session_id):
        """Generate unique hash ID for each chunk"""
        unique_string = text + session_id
        return hashlib.md5(unique_string.encode()).hexdigest()

    def add(self, embeddings, documents, metadatas):
        print("📥 Checking for new documents to add...")

        existing_data = self.collection.get()
        existing_ids = set(existing_data["ids"]) if existing_data["ids"] else set()

        new_embeddings = []
        new_documents = []
        new_ids = []
        new_metadatas = []

        for doc, emb, meta in zip(documents, embeddings, metadatas):
            doc_id = self.generate_id(doc, meta["session_id"])  # 🔥 FIX HERE

            if doc_id not in existing_ids:
                new_documents.append(doc)
                new_embeddings.append(emb)
                new_ids.append(doc_id)
                new_metadatas.append(meta)

        if len(new_ids) == 0:
            print("⚡ No new documents to add.")
            return

        print(f"+ Adding {len(new_ids)} new chunks...")

        self.collection.add(
            embeddings=new_embeddings,
            documents=new_documents,
            ids=new_ids,
            metadatas=new_metadatas
        )

        print("✅ New embeddings stored successfully.")
        
    def search(self, query_embedding, k=3, session_id=None):
        """
        Search top-k similar documents
        Returns both documents and metadata
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"session_id": session_id}
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else []
        }