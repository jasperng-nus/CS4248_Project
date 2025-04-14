import os
from openai import OpenAI
import pandas as pd
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from retrieval import retrieve

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"
SIMILARITY_SCORE_THRESHOLD = 0.2
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query, top_k=5):

    results = retrieve(query, top_k)
    generated_answer = determine_model_output(results)
    return generated_answer

def determine_model_output(results):

    answer = ""
    similarityScoreList = results["similarity_score"].tolist()
    
    if all(score < SIMILARITY_SCORE_THRESHOLD for score in similarityScoreList):
        prompt = (
            f"Answer the following question generally: \n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        answer += "The requested information is not present in our database. As such our RAG will provide a general response:\n\n"
    else:
        filtered_results = results[results["similarity_score"] >= 0.2]
        context = filtered_results["text_for_embeddings"].tolist()
        context_chunk = "\n".join(context)
        prompt = (
            f"Use the following context to answer the question. \n\n"
            f"Context:\n{context_chunk}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        citingPaperIdList = filtered_results["citingPaperId"].tolist()
        claimList = filtered_results["string"].tolist()
        citations = "\n\nThe RAG has referenced claims from the following papers:\n" 
        for index, citingPaperId in enumerate(citingPaperIdList):
            citations += f"{index+1}. {claimList[index]} Obtained from Citing Paper ID: {citingPaperId}. (Similarity Score: {similarityScoreList[index]})\n\n"
        answer += citations

    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )
    answer += response.output_text
    return answer


if __name__ == "__main__":
    # query = "What are the effects of climate change on coral reefs?"
    query = "What were Mouse embryonic fibroblasts (MEFs) infected with?"
    print(generate_answer(query))
