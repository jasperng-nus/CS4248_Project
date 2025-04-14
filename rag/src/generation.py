import os
from openai import OpenAI
import pandas as pd
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from retrieval import retrieve

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query, top_k=5):

    results = retrieve(query, top_k)
    context = results["text_for_embeddings"].tolist()
    context_chunk = "\n".join(context)
    prompt = (
        f"Use the following context to answer the question. \n\n"
        f"Context:\n{context_chunk}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )

    generated_answer = response.output_text

    citingPaperIdList = results["citingPaperId"].tolist()
    claimList = results["string"].tolist()
    citations = "\n\nThe RAG has referenced claims from the following papers:\n" 
    for index, citingPaperId in enumerate(citingPaperIdList):
        citations += f"{index+1}. {claimList[index]} ({citingPaperId})\n"

    generated_answer += citations

    return generated_answer

if __name__ == "__main__":
    query = "What are the effects of climate change on coral reefs?"
    print(generate_answer(query))