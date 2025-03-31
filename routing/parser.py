from abc import ABC, abstractmethod
import re


class Parser(ABC):
    @abstractmethod
    def parse(self, question):
        """
        Parse the question and return a structured representation.
        """
        pass

class LLMParser(Parser):

    def parse(self, response):
        """
        Parse the response and return a structured representation.
        """
        pattern = "Classification: (Background|Method|Result)"
        match = re.search(pattern, response)
        if match:
            classification = match.group(1)
            return classification
        else:
            raise ValueError("Response does not contain a valid classification.")
        


        