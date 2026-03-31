import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query, context_chunks):
    cleaned_chunks = list(set(context_chunks))[:3]
    context = "\n\n".join(cleaned_chunks)

    prompt = f"""
        You are an intelligent assistant helping students understand university documents.

        Use the provided context to answer the question.

        If the answer is found in the context:
        - Explain clearly in simple language
        - Summarize in 3–5 concise bullet points
        - Avoid repeating the same information
        - Merge similar points into one
        - Be crisp and non-redundant

        If the answer is partially available:
        - Answer using what is available
        - Do NOT use outside knowledge
        - If answer is not in context → say "I could not find this in the uploaded documents"
        - Keep answer concise and clear

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    response = llm.invoke(prompt)
    return response.content.strip()

def rewrite_query(query, history):
    """
    Convert follow-up question into standalone question
    """

    if not history:
        return query

    last_item = history[-1]

    # Handle different formats safely
    last_q = (
        last_item.get("q")
        or last_item.get("question")
        or last_item.get("query")
        or ""
    )

    if not last_q:
        return query

    prompt = f"""
    You are a helpful assistant.

    Rewrite the current question so it is fully self-contained.

    Previous question:
    {last_q}

    Current question:
    {query}

    Rewritten question:
    """

    rewritten = generate_answer(prompt, [])

    return rewritten.strip() if rewritten else query