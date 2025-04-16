import os
from openai import OpenAI
import pandas as pd
import sys
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from rag.src.retrieval import retrieve
from llm.llm import LLMEngine


GENERATION_MODEL = "gpt-4o-mini"
SIMILARITY_SCORE_THRESHOLD = 0.2
client = LLMEngine()


def get_system_message():
    system_message = (
            "You are a helpful scientific query assistant."
            "You will be provided with a question and a set of scientific papers. "
            "Your task is to answer the question based on the provided papers. "
            "If the information is not present in the papers, provide a general response. "
            "If the information is present, synthesize the relevant content from the papers to provide a detailed, evidence-based answer."
            "Be concise and clear in your response. "
            "Example 1:\n"
            "Question: What is the effect of climate change on coral reefs?\n"
            "Context: Climate change has a significant impact on coral reefs, including coral bleaching, ocean acidification, and changes in species composition. These effects can lead to the decline of coral reef ecosystems. This means that it is important that corals have a sustainable lifestyle.\n\n"
            "Answer: Climate change has a significant impact on coral reefs, including coral bleaching, ocean acidification, and changes in species composition. These effects can lead to the decline of coral reef ecosystems.\n\n"
            "Example 2: \n"
            "Question: What are the colours of the rainbow?\n"
            "Answer: The colors of the rainbow are Violet, Indigo, Blue, Green, Yellow, Orange and Red.\n\n"
    )
    return system_message


def generate_answer(query, top_k=5, rag=True, testMode = False):
    results = retrieve(query, top_k)
    generated_answer = determine_model_output(results, query, testMode) if rag else determine_model_output_norag(query)
    return generated_answer

def determine_model_output(results, query, testMode):
    system_message = get_system_message()
    human_message = ""
    answer = ""
    similarityScoreList = results["similarity_score"].tolist()
    
    if all(score < SIMILARITY_SCORE_THRESHOLD for score in similarityScoreList):
        human_message = (
            f"Question: {query}\n"
            "Answer:"
        )
        if testMode:
            pass
        else:
            answer += "The requested information is not present in our database. As such our RAG will provide a general response:\n\n"
    else:
        context, filtered_results = retrieve_similar_questions_and_context(results)
        context_chunk = "\n".join(context)
        human_message = (
            f"Context:\n{context_chunk}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        citations = retrieve_citations(filtered_results, similarityScoreList)
        if testMode:
            pass
        else:
            answer += citations

    response = client.generate_response([
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message}
    ])

    answer += response
    return answer

def retrieve_similar_questions_and_context(results):
    filtered_results = results[results["similarity_score"] >= 0.2]
    context = filtered_results["text_for_embeddings"].tolist()
    return context, filtered_results

def retrieve_citations(filtered_results, similarityScoreList):
    citingPaperIdList = filtered_results["citingPaperId"].tolist()
    claimList = filtered_results["string"].tolist()
    citations = "\n\nThe RAG has referenced claims from the following papers:\n" 
    for index, citingPaperId in enumerate(citingPaperIdList):
        citations += f"{index+1}. {claimList[index]} (Claim obtained from Citing Paper ID: {citingPaperId}. Similarity Score: {similarityScoreList[index]:.2f})\n\n"
    return citations

def determine_model_output_norag(query):

    answer = ""
    system_message = get_system_message()
    human_message = (
        f"Question: {query}\n"
        "Answer:"
    )
    response = client.generate_response([
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message}
    ])
    answer += response
    return answer

if __name__ == "__main__":
    # query = "What are the effects of climate change on coral reefs?"
    # query = "What were Mouse embryonic fibroblasts (MEFs) infected with?"
    query = "Among the subgroups with minor depressive symptoms at baseline ( CES-D score 16-20 ) , did the CES-D score increase of decrease?"
    print(generate_answer(query))
