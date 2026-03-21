from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.loader import load_documents
from app.utils.chunker import chunk_text
from app.services.embedding import get_embeddings
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer

#Loading Pdfs
docs = load_documents("data")

all_chunks = []

for doc in docs:
    chunks = chunk_text(doc["content"])
    
    for chunk in chunks:
        all_chunks.append(f"[{doc['filename']}] {chunk}")
        
print(f"Total Chunks: {len(all_chunks)}")

#Generating Embeddings
embeddings = get_embeddings(all_chunks)

#Creating Vector Store
vector_store = VectorStore()
vector_store.add(embeddings,all_chunks)

#Testing
query = "what if i loose my form I-20? how many days do i have to wait?"
query_embedding = get_embeddings([query])[0]

results = vector_store.search(query_embedding, k= 3)
 
answer = generate_answer(query, results)
print("\n Final Answer: \n")
print(answer)
