from llm.llm import LLMEngine
from routing.messages import Messages
from routing.parser import LLMParser

class Router:
    def __init__(self, question):
        self.question = question
        self.llm = LLMEngine()
        self.parser = LLMParser()

    def route(self):
        messages = Messages(self.question).messages
        response = self.llm.generate_response(messages)
        parsed_response = self.parser.parse(response)
        return parsed_response