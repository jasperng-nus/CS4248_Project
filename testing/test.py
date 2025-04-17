import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rag.src.generation import generate_answer
from llm.llm import LLMEngine
from testing.messages import Messages

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_sample(sample, llm):
    question = sample['question']
    answer = sample['answer']

    response_norag = generate_answer(question, rag=False, testMode=True)
    response_rag = generate_answer(question, rag=True, testMode=True)

    verdict_norag = llm.generate_response(Messages(question, response_norag, answer).messages)
    verdict_rag = llm.generate_response(Messages(question, response_rag, answer).messages)
    return {
        "question": question,
        "answer": answer,
        "response_norag": response_norag,
        "response_rag": response_rag,
        "verdict_norag": verdict_norag,
        "verdict_rag": verdict_rag,
        "accuracy_norag": "True" in verdict_norag,
        "accuracy_rag": "True" in verdict_rag
    }


def main():
    llm = LLMEngine()

    outputs = []
    response_accuracy = []
    response_accuracy_augmented = []
    data_path = os.path.join(project_root, 'testing', 'data.json')
    with open(data_path, 'r') as file:
        data = json.load(file)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_sample, sample, llm) for sample in data]

        for count, future in enumerate(as_completed(futures)):
            if count % 20 == 0:
                print(count)
            try:
                result = future.result()
                outputs.append(result)
                response_accuracy.append(result["accuracy_norag"])
                response_accuracy_augmented.append(result["accuracy_rag"])
            except Exception as e:
                print(f"Error processing sample: {e}")

    print(f"Unaugmented LLM Accuracy = {(sum(response_accuracy) / len(response_accuracy))}")
    print(f"Augmented LLM Accuracy = {(sum(response_accuracy_augmented) / len(response_accuracy_augmented))}")

    with open("outputs.json", "w") as f:
        json.dump(outputs, f, indent=4)
        
if __name__ == "__main__":
    main()
    