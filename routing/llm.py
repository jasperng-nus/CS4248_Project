import os
from openai import OpenAI


class LLM:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(self, prompt):
        out = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        )
        response = out.choices[0].message.content
        return response
       
