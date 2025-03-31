class Messages:
    def __init__(self, question):
        self.user_message = f"Question: {question}\nReasoning: "
        self.system_message = """
You are an AI assistant tasked with classifying a question into one of three sections of a research paper: 
'Background', 'Method', or 'Result'. 

Here are the definitions of the sections:
1. **Background**: This section provides context, prior work, and foundational information about the research.
   Questions about general concepts, motivations, or related work belong here.
2. **Method**: This section describes the experimental setup, algorithms, or procedures used in the research.
   Questions about how something was done or implemented belong here.
3. **Result**: This section presents the outcomes, findings, or conclusions of the research.
   Questions about specific results, data, or interpretations belong here.

Follow these steps:
1. Analyze the question to understand its focus.
2. Compare the focus of the question with the definitions of the sections.
3. Decide which section is most relevant and explain your reasoning.

Here are some examples:

Question: What causes earthquakes?
Reasoning: This question seeks a general scientific explanation of a natural phenomenon rather than experimental methods or specific findings.
Classification: Background

Question: How does photosynthesis work?
Reasoning: The question asks for an explanation of a well-established biological process, making it part of foundational knowledge.
Classification: Background

Question: What are the main factors influencing climate change?
Reasoning: The question asks for broad scientific knowledge about environmental changes rather than methods or specific results.
Classification: Background

Question: How do scientists measure air pollution levels?
Reasoning: This question focuses on the techniques and methodologies used to gather environmental data.
Classification: Method

Question: What techniques are used to date ancient fossils?
Reasoning: It asks about scientific procedures for determining the age of fossils rather than the findings themselves.
Classification: Method

Question: What statistical tests are commonly used in medical research?
Reasoning: The question concerns the statistical techniques used to analyze medical data rather than specific study results.
Classification: Method

Question: How much has the Earth’s temperature increased in the last century?
Reasoning: This question seeks specific numerical data on global warming trends, which falls under scientific results.
Classification: Result

Question: What is the success rate of organ transplants?
Reasoning: The question asks for statistical findings from medical studies rather than the methodology or background concepts.
Classification: Result

Question: What has research shown about the link between diet and mental health?
Reasoning: The question asks for established conclusions drawn from scientific studies rather than theoretical explanations.
Classification: Result

Now, classify the given question using the same reasoning process.
"""
        self.messages = [
            {"role": "user", "content": self.user_message},
            {"role": "system", "content": self.system_message},
        ]

