import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query, context_chunks):
    cleaned_chunnks = list(set(context_chunks))[:3]
    context = "\n\n".join(cleaned_chunnks)
    
    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    IMPORTANT RULES:
    - Give a clear and concise answer
    - Do NOT repeat the context
    - Do NOT include unnecessary details
    - Summarize the answer in 3–5 bullet points if possible
    
    Context:
    {context}

    Question:
    {query}

    Final Answer:
    """
    
    response = llm.invoke(prompt)
    
    return response.content.strip()