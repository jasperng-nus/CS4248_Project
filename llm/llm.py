import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

class LLMEngine:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(self, prompt):
        out = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.7,
        )
        response = out.choices[0].message.content
        return response
    
    def generate_embeddings(self, query_text, model="text-embedding-3-small"):
        response = self.client.embeddings.create(
            input=query_text,
            model=model,
        )
        return response
            
       
