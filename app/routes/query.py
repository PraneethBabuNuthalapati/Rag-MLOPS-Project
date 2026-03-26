from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embedding import get_embeddings
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer
from app.services.reranker import rerank
from app.services.llm import rewrite_query

router = APIRouter(prefix="/api", tags=["Query"])


class QueryRequest(BaseModel):
    query: str
    session_id: str
    history: list = []  # Optional: can be used for more advanced context handling in the future


@router.post("/query")
def query_rag(req: QueryRequest):

    final_query = rewrite_query(req.query, req.history)
    
    query_embedding = get_embeddings([final_query])[0]
    vs = VectorStore()

    # Filter at DB level
    results = vs.search(
        query_embedding,
        k=6,
        session_id=req.session_id
    )

    documents = results["documents"]
    metadatas = results["metadatas"]

    if not documents:
        return {
            "query": final_query,
            "answer": "I could not find this in the uploaded documents.",
            "sources": []
        }

    #  Rerank
    reranked = rerank(final_query, documents, top_k=5)

    # remove duplicates
    seen = set()
    retrieved_chunks = []

    for chunk in reranked:
        if chunk not in seen:
            retrieved_chunks.append(chunk)
            seen.add(chunk)

    retrieved_chunks = retrieved_chunks[:3]

    if not retrieved_chunks:
        return {
            "query": final_query,
            "answer": "I could not find this in the uploaded documents.",
            "sources": []
        }

    #  Generate answer
    answer = generate_answer(final_query, retrieved_chunks)

    #  Clean sources
    clean_sources = []
    for chunk in retrieved_chunks:
        if "]" in chunk:
            file_name = chunk.split("]")[0].replace("[", "")
            snippet = chunk.split("]")[1].strip()[:150]
        else:
            file_name = "unknown"
            snippet = chunk[:150]

        clean_sources.append({
            "file": file_name,
            "snippet": snippet
        })

    return {
        "query": req.query,
        "rewritten_query": final_query,
        "answer": answer,
        "sources": clean_sources
    }