import ollama
import openai
import re
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.schema.messages import SystemMessage
from langsmith import Client
from PIL import Image
import torch
import numpy as np
from transformers import AutoTokenizer
import traceback

from dotenv import find_dotenv, load_dotenv

from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool
from typing import List, Dict, Any

from langchain_openai import AzureOpenAI

from langsmith.wrappers import wrap_openai

import os

import pprint

from langchain_community.utilities import SearxSearchWrapper

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('ENDPOINT')
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('API_KEY')

# import nest_asyncio

# nest_asyncio.apply()

# client = Client(api_key="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", api_url="https://api.smith.langchain.com")

# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langsmith-tutorial"

#embedding = OllamaEmbeddings(model="nomic-embed-text")

import tiktoken

model_name = "gpt-4"

encoding = tiktoken.encoding_for_model(model_name)

def get_tiktoken_length(text):
    return len(encoding.encode(text))


# Set environment variables
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT

# Initialize the embedding model
embedding = AzureOpenAIEmbeddings(
#    deployment="your-deployment-name",  # Replace with your actual deployment name
    model="text-embedding-3-large",
    dimensions=1536,
#    api_version="2024-02-15-preview",
    chunk_size=512,
)

db_course_name = Chroma(persist_directory="chroma_db_course_name", embedding_function=embedding)

retriever_course_name = db_course_name.as_retriever()

db_course_overview = Chroma(persist_directory="chroma_db_course_overview", embedding_function=embedding)

retriever_course_overview = db_course_overview.as_retriever()

db_course_notes = Chroma(persist_directory="chroma_db_notes", embedding_function=embedding)

retriever_course_notes = db_course_notes.as_retriever()

db_course_textbook = Chroma(persist_directory="chroma_db_textbook", embedding_function=embedding)

retriever_course_textbook = db_course_textbook.as_retriever()

db_kc = Chroma(persist_directory="chroma_db_kc", embedding_function=embedding)

retriever_kc = db_kc.as_retriever()

db_course_logistics = Chroma(persist_directory="course-logistics-retriever", embedding_function=embedding)

retriever_course_logistics = db_course_logistics.as_retriever()

# search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888", k=5)

tools = [
    create_retriever_tool(
        retriever_course_name,
        "search_course_name",
        "This includes just the course name.",
    ),
    create_retriever_tool(
        retriever_course_overview,
        "search_course_overview",
        "Search for course overview. This includes the course overview and policies.",
    ),
    create_retriever_tool(
        retriever_course_notes,
        "search_course_notes",
        "Search for information or answers from the course notes. This includes content and concepts related to the course This should be your primary choice for answering anny course-specific questions.",
    ),
    create_retriever_tool(
        retriever_course_textbook,
        "search_course_textbook",
        "Search for information or answers from the course textbook. This includes content related to C programming and should be your first choice to answer any C programing related questions.",
    ),
    create_retriever_tool(
        retriever_kc,
        "search_knowledge_components",
        "Search for relevant Knowledge Components (KCs) in C programming. Use this tool to identify specific areas of knowledge that the student might need to focus on for debugging and problem-solving.",
    ),
    create_retriever_tool(
        retriever_course_logistics,
        "search_course_logistics",
        "Search for information about course logistics, schedules, policies, and other administrative details from the Canvas course page and related links. Use it to provide exact answers to any questions the student will have regarding the course itself.",
    ),
]

# llm = OllamaLLM(model="llama3.1")

# llm = AzureOpenAI(
#     azure_endpoint=ENDPOINT,
#     api_key=API_KEY,
#     api_version="2024-02-15-preview"
# )

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt4-o",  # or your deployment
    api_version="2024-02-15-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# system_message = """You are a ECE 120 AI teaching assistant who provides provides a helpful, and informative hints and answers unless it is a debugging or code question the the student asks and provides specific guidance to the student to learn more about the topic. Never the direct answer to the student when a question about debugging is asked but rather provide hints to guide the students in the right path. 
#   You can use the following tools to look up relevant information:
# - Search for course name. This includes just the course name and often the content and may not be relevant to your question.
# - Search for course overview. This includes only the course overview and policies and sometimes this may not be relevant to your question.
# - Search for information or answers from the course notes. If answer found here, return the section of the course notes you found the answer in, else check the textbook.
# - Search for information or answers from the course textbook. If answer found here, return the section of the textbook you found the answer in.
# """

message = """You are the no-nonsense AI teaching assistant for ECE 120: Introduction to Computing. Your job is to guide students to answers, not spoon-feed them solutions. Adhere to these rules:

1. NEVER provide direct code solutions or debugging fixes. Instead, offer conceptual explanations and point students to relevant resources.

2. Use a tone that's direct, slightly intimidating and joking, and occasionally sarcastic - like a real ECE TA.

3. Encourage independent thinking. Push students to derive answers themselves.

4. For any question, use these information sources in order:
   a) Course notes search
   b) Textbook search
   c) Course name search
   d) Course overview search
   e) Course logistics serarch
   Always cite the specific section and page number of your source.

5. If a student asks about course policies or logistics, refer them to the course overview.

6. For conceptual questions, provide clear, detailed and helpful explanations with relevant examples from the course material.

7. If a student is struggling, break down the problem into smaller steps and guide them through the thought process.

8. When helping with debugging, follow these steps:
   a) Identify the relevant Knowledge Components for the problem at hand.
   b) Ask the student targeted questions about specific KCs to pinpoint their understanding.
   c) Based on their responses, provide hints and explanations that focus on the relevant KCs.
   d) Encourage the student to apply the KC-specific knowledge to debug their code.

9. If more information is needed about the student's code, ask specific questions related to the relevant KCs. For example:
   - "Can you show me the function where you're experiencing the issue?"
   - "What data types are you using for your variables in this section?"
   - "Have you checked for proper memory allocation and deallocation?"
   
10. When providing hints, relate them to specific Knowledge Components:
    - Syntax and Structure: "Review the syntax for [specific construct]. Are all your brackets and semicolons in the right places?"
    - Memory Management: "Consider how you're allocating memory for this data structure. Are you freeing all allocated memory?"
    - Data Types and Operations: "Think about the data types you're using. Are they appropriate for the operations you're performing?"
    - Input/Output: "Check your I/O functions. Are you handling all possible input cases?"
    - Debugging Techniques: "Try adding print statements before and after this section to track variable values."
    - Code Organization: "Consider breaking this function into smaller, more manageable parts."

11. Foster metacognitive skills by asking students to reflect on their problem-solving process:
    - "What debugging steps have you taken so far?"
    - "How did you approach solving this problem initially?"
    - "What resources have you consulted before asking for help?"
    
12. Adjust the level of hints based on the student's demonstrated understanding of relevant KCs:
    - For beginners: Provide more detailed explanations and step-by-step guidance.
    - For intermediate learners: Offer more targeted hints and encourage independent problem-solving.
    - For advanced students: Challenge them with thought-provoking questions and minimal hints.

13. When debugging or problem-solving, use the Knowledge Components Search tool to identify relevant areas of C programming knowledge. This tool will help you provide more targeted assistance based on the specific concepts involved in the student's question or issue.

14. After using the Knowledge Components Search tool, incorporate the retrieved information into your response. Explain how the identified knowledge components relate to the student's problem and guide them towards applying this knowledge.

Remember, your goal is to make students think, not to make their lives easier. Now go forth and toughen up these future engineers!
"""

# #Example:
# User: Implement the function int isPrime(int n) in C.
# AI: Start by defining the function signature in C. The function signature would look like this: int isPrime(int n) { ... }.
#         In the course notes for ECE 120, you can refer to the following sections to learn more about C programming:

#         Chapter 1.5: Programming Concepts and the C Language - This section introduces the C programming language and explains basic concepts in computer programming. It covers topics such as variables, operators, functions, statements, and program execution. (Page 23-30)
#         You can also refer to the textbook "Introduction to Computing Systems: From Bits and Gates to C and Beyond" by Yale N. Patt and Sanjay J. Patel. Here are some relevant sections from the textbook:

#         Chapter 11: Introduction to C/C++ Programming - This chapter introduces fundamental high-level programming constructs in C and C++. It covers variables, control structures, functions, arrays, pointers, recursion, and simple data structures. It also provides a problem-solving methodology for programming. (Page 406-450)
# """

system_message = SystemMessage(content=message)

# def process_image_query(model, tokenizer, image, query):
#     # Convert image to RGB
#     image = image.convert('RGB')
    
#     # Prepare the message context
#     msgs = [{"role": "user", "content": query}]
    
#     # Set default parameters
#     params = {
#         'sampling': True,
#         'top_p': 0.8,
#         'top_k': 100,
#         'temperature': 0.7,
#         'repetition_penalty': 1.05,
#         'max_new_tokens': 896
#     }
    
#     ERROR_MSG = "Error, please retry"
    
#     # Generate the response
#     try:
#         answer = model.chat(
#             image=image,
#             msgs=msgs,
#             tokenizer=tokenizer,
#             **params
#         )
        
#         # Clean up the response
#         res = re.sub(r'(.*)', '', answer)
#         res = res.replace('', '')
#         res = res.replace('', '')
#         res = res.replace('', '')
#         answer = res.replace('', '')
        
#         return answer
    
#     except Exception as err:
#         print(err)
#         traceback.print_exc()
#         return ERROR_MSG
    
def get_rag_agent():
    class RunIDCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self.run_id = None

        def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
            self.run_id = kwargs.get("run_id")
            
    # Create callback handlers
    stdout_handler = StdOutCallbackHandler()
    run_id_handler = RunIDCallbackHandler()

    model_name = "gpt-4"
    encoding = tiktoken.encoding_for_model(model_name)

    def get_tiktoken_length(text):
        return len(encoding.encode(text))
    
    llm.model_name = "gpt-4o"
    
    agent = create_conversational_retrieval_agent(
        llm,
        tools,
        system_message=system_message,
        remember_intermediate_steps=True,
        verbose=True,
        callback_manager=None,
    )
    
    return agent