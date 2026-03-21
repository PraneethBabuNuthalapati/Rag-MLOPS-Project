import logging
import os
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

def load_documents(folder_path):
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path,file))
            text = ""
            
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text+=content.replace("\n", " ")
            
            documents.append({
                "filename": file,
                "content": text
            })
            
    return documents
