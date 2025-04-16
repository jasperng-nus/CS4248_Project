import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rag.src.generation import generate_answer
from llm.llm import LLMEngine
from testing.messages import Messages

def main():
    llm = LLMEngine()

    outputs = []
    response_accuracy = []
    response_accuracy_augmented = []

    with open('data.json', 'r') as file:
        data = json.load(file)

    count = 0
    for sample in data:
        if count%20==0:
            print(count)
        question = sample['question']
        answer = sample['answer']

        response_norag = generate_answer(question, rag=False, testMode=True)
        response_rag = generate_answer(question, rag=True, testMode=True)

        ## response_rag can have unnecessary information about the RAG has refenced... We do not care about those
        ## we will extract just the string answer

        verdict_norag = llm.generate_response(Messages(question, response_norag, answer).messages)
        verdict_rag = llm.generate_response(Messages(question, response_rag, answer).messages)

        response_accuracy.append("True" in verdict_norag)
        response_accuracy_augmented.append("True" in verdict_rag)

        outputs.append({
            "question": question,
            "answer": answer,
            "response_norag": response_norag,
            "response_rag": response_rag,
            "verdict_norag": verdict_norag,
            "verdict_rag": verdict_rag
        })
        count+=1


    print(f"Unaugmented LLM Accuracy = {(sum(response_accuracy) / len(response_accuracy))}")
    print(f"Augmented LLM Accuracy = {(sum(response_accuracy_augmented) / len(response_accuracy_augmented))}")

    with open("outputs.json", "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    main()
    