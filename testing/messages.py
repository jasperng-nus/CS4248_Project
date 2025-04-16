class Messages:
    def __init__(self, question, response, model_ans):
        self.user_message = f"Question: {question}\nResponse: {response}\nModel Answer: {model_ans}\nVerdict:"
        self.system_message = """
You are an AI assistant tasked with determining whether a generated response is factually equivalent to a Model Answer.

Follow these steps:
1. Read and analyze the Model Answer first. Understand the key concept or fact being conveyed.
2. Read and analyze the Response. 
3. Determine whether the Response **conveys the same factual meaning** as the Model Answer.
    - Ignore differences in wording, phrasing, or style.
    - Focus on whether the Response accurately captures the essence of the Model Answer.
    - If the Response is a paraphrase or rephrasing of the Model Answer, it is considered equivalent as well.
4. Only if the Response is **semantically consistent** and **does not hallucinate**, output "True". Otherwise, output "False".

Here are some examples:

Question: What mechanism does contractile agonists cause smooth muscle cells to generate force by?
Response: Contractile agonists cause smooth muscle cells to generate force primarily through the calcium-calmodulin-MLCK pathway.
Model Answer: The actomyosin mechanism.
Verdict: False

Question: What mechanism does contractile agonists cause smooth muscle cells to generate force by?
Response: Contractile agonists cause smooth muscle cells to generate force primarily through the actomyosin mechanism, which involves interactions between actin and myosin filaments.
Model Answer: The actomyosin mechanism.
Verdict: True
"""
        self.messages = [
            {"role": "user", "content": self.user_message},
            {"role": "system", "content": self.system_message},
        ]

