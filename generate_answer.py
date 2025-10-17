import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """You are an expert assistant. Using the following SOURCES from documents, answer the question concisely and cite the source names where applicable.

SOURCES:
{sources}

QUESTION:
{question}

Answer succinctly, and if the answer is not available in the documents, say "Not enough information in the provided documents."."""

def synthesize_answer(question, retrieved_chunks, max_tokens=250, model="gpt-3.5-turbo"):
    # build the sources text
    sources_txt = ""
    for i, r in enumerate(retrieved_chunks):
        src = r["metadata"].get("source","unknown")
        chunk_text = r["text"].strip().replace("\n"," ")
        sources_txt += f"[{i}] Source: {src}\n{chunk_text}\n\n"

    prompt = PROMPT_TEMPLATE.format(sources=sources_txt, question=question)
    # Use chat completion
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a helpful assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        answer = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"LLM error: {e}"
    return answer
