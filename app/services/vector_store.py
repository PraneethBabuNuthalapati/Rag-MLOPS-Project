import chromadb
import hashlib


class VectorStore:
    def __init__(self, persist_dir="chroma_db", collection_name="rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def generate_id(self, text):
        """Generate unique hash ID for each chunk"""
        return hashlib.md5(text.encode()).hexdigest()

    def add(self, embeddings, documents):
        """
        Add only NEW documents (no duplicates)
        Works for both static data and user uploads
        """

        print("📥 Checking for new documents to add...")

        # Get existing IDs
        existing_data = self.collection.get()
        existing_ids = set(existing_data["ids"]) if existing_data["ids"] else set()

        new_embeddings = []
        new_documents = []
        new_ids = []

        for doc, emb in zip(documents, embeddings):
            doc_id = self.generate_id(doc)

            # Only add if not already present
            if doc_id not in existing_ids:
                new_documents.append(doc)
                new_embeddings.append(emb)
                new_ids.append(doc_id)

        if len(new_ids) == 0:
            print("⚡ No new documents to add.")
            return

        print(f"➕ Adding {len(new_ids)} new chunks...")

        self.collection.add(
            embeddings=new_embeddings,
            documents=new_documents,
            ids=new_ids
        )

        print("✅ New embeddings stored successfully.")

    def search(self, query_embedding, k=3):
        """Search top-k similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results["documents"][0]