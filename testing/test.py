import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rag.src.generation import generate_answer
from routing.llm import LLM
from testing.messages import Messages

def main():
    llm = LLM()

    outputs = []
    response_accuracy = []
    response_accuracy_augmented = []

    with open('data.json', 'r') as file:
        data = json.load(file)

    for sample in data:
        print(sample)
        question = sample['question']
        answer = sample['answer']

        response_norag = generate_answer(question, rag=False)
        response_rag = generate_answer(question, rag=True)

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


    print(f"Unaugmented LLM Accuracy = {(sum(response_accuracy) / len(response_accuracy))}")
    print(f"Augmented LLM Accuracy = {(sum(response_accuracy_augmented) / len(response_accuracy_augmented))}")

    with open("outputs.json", "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    main()
    