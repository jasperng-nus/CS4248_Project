class Messages:
    def __init__(self, question, response, model_ans):
        self.user_message = f"Question: {question}\nResponse: {response}\nModel Answer: {model_ans}\nVerdict:"
        self.system_message = """
You are an AI assistant tasked with deciding if a response matches with a Model Answer.

Follow these steps:
1. Analyze the model answer, followed by the responses.
2. Compare the response and the model answer if they mean the same thing.
3. If the response matches the model answer, answer "True", otherwise, answer "False"

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

