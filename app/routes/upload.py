from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import uuid

from app.utils.loader import load_pdf
from app.utils.chunker import chunk_text
from app.services.embedding import get_embeddings
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["Upload"])

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(..., description="Upload PDF files")):
    
    session_id = str(uuid.uuid4())
    all_chunks = []
    all_metadata = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load PDF
        docs = load_pdf(UPLOAD_DIR)

        for doc in docs:
            if doc["filename"] == file.filename:
                chunks = chunk_text(doc["content"])

                for chunk in chunks:
                    # prepend filename for traceability
                    full_chunk = f"[{file.filename}] {chunk}"

                    all_chunks.append(full_chunk)

                    all_metadata.append({
                        "filename": file.filename,
                        "session_id": session_id
                    })

    # Embed + store
    batch_size = 32
    embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        embeddings.extend(get_embeddings(batch))

    vs = VectorStore()
    vs.add(embeddings, all_chunks, all_metadata)
    

    return {
        "message": "Files uploaded successfully",
        "session_id": session_id,
        "num_chunks": len(all_chunks)
    }