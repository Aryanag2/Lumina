# import openai
# import gradio as gr
# from rag_agent import llm, get_rag_agent
# import logging
# import uuid
# import os
# import sys
# import traceback
# from jdoodle import compile_c_code
# from transformers import AutoModel, AutoTokenizer
# import torch
# from 
# 
# import Client
# from langchain.callbacks import StdOutCallbackHandler
# from langchain_core.callbacks import BaseCallbackHandler
# from langchain_core.language_models import BaseLanguageModel
# from langchain.tools import BaseTool
# from typing import List, Dict, Any
# from db import authenticate_user, register_user, log_chat, get_chat_history, get_user_by_session

# from openai import AzureOpenAI

# class RunIDCallbackHandler(BaseCallbackHandler):
#     def __init__(self):
#         self.run_id = None
#     def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
#         self.run_id = kwargs.get("run_id")

# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langsmith-tutorial"

# print("API-KEY: " + os.environ.get("LANGCHAIN_API_KEY"))
# print("ENDPOINT: " + os.environ.get("LANGCHAIN_ENDPOINT"))

# print(openai.__version__)

# client = Client()

# def get_initial_prompt():
#     return """🤖: Hello!
# I am an AI Teaching Assistant built to answer all your questions on ECE120.
# - You can ask about ECE120, any assignment, and even general concepts such as assembly.
# - You can ask about specific concepts such as "What is a logic gate?"
# - Seek clarification on course material: "Can you explain the concept of state diagrams with an example?"
# - Inquire about assignment instructions: "What should I do if my program doesn't compile?"
# - Remember, the more specific your question, the better I can assist you!"""

# # Load the MiniCPM-Llama3-V-2_5 model
# # model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, device_map='auto')
# # model.eval()

# # tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)

# def build_app():
#     # Session and agent states
#     session_state = gr.State({"logged_in": False, "session_id": None})
#     agent_state = gr.State(None)
#     run_id = gr.State(None)
    
#     with gr.Blocks() as demo:
#         gr.HTML("""<h1 align="center">🤖: ECE120 AI-TA</h1>""")
        
#         # Define both interfaces at the top level
#         login_interface = gr.Group(visible=lambda: not session_state.value["logged_in"])
#         assistant_interface = gr.Group(visible=lambda: session_state.value["logged_in"])
        
#         # Login Interface
#         with login_interface:
#             gr.Markdown("## Please Log In or Register")
#             username = gr.Textbox(label="Username", placeholder="Enter your username")
#             password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
#             login_btn = gr.Button("Login")
#             register_btn = gr.Button("Register")
#             login_output = gr.Textbox(label="Status", interactive=False)
            
#             # Login function
#             def login(username_input, password_input):
#                 session_id_value = authenticate_user(username_input, password_input)
#                 if session_id_value:
#                     session_state.value["logged_in"] = True
#                     session_state.value["session_id"] = session_id_value
#                     # Initialize the agent when the user logs in
#                     agent_state.value = get_rag_agent()
#                     return "Login successful!", gr.update(visible=False), gr.update(visible=True)
#                 else:
#                     return "Login failed. Please try again.", gr.update(), gr.update()
            
#             # Register function
#             def register(username_input, password_input):
#                 message = register_user(username_input, password_input)
#                 return message
            
#             login_btn.click(
#                 login,
#                 inputs=[username, password],
#                 outputs=[login_output, login_interface, assistant_interface]
#             )
#             register_btn.click(register, inputs=[username, password], outputs=[login_output])
        
#         # Assistant Interface
#         with assistant_interface:
#             with gr.Row():
#                 with gr.Column(scale=3):
#                     initial_prompt = get_initial_prompt()
#                     chatbot = gr.Chatbot(value=[("AI Teaching Assistant", initial_prompt)], height=600, min_width=300, bubble_full_width=False)
#                     with gr.Row():
#                         msg = gr.Textbox(placeholder="Insert question here...", scale=4)
#                         image_input = gr.Image(type="pil", label="Upload Image (optional)", scale=1)
                    
#                     starter_code = """#include <stdio.h>\nint main() \n{\n    printf("Hello World");\n    return 0;\n}"""
#                     code_editor = gr.TextArea(value=starter_code, label="C Code Editor", lines=10)
#                     compile_button = gr.Button("Compile")
#                     code_output = gr.TextArea(placeholder="Output will appear here...", label="Compiler Output", lines=10)
#                     clear = gr.ClearButton([msg, chatbot, code_editor, code_output])
#                     with gr.Row():
#                         feedback_text = gr.Textbox(label="Feedback (optional)")
#                         feedback_score = gr.Slider(minimum=0, maximum=1, step=0.1, label="Score")
#                         submit_feedback = gr.Button("Submit Feedback")
#                     logout_btn = gr.Button("Logout")

#             # Define the respond function
#             # def respond(message, chat_history, image, request: gr.Request):
#             #     try:
#             #         if not session_state.value["logged_in"]:
#             #             return "Please log in to continue.", chat_history, None

#             #         full_context = " ".join([f"User: {msg} AI: {resp}" for msg, resp in chat_history])
#             #         full_context += f" User: {message}"

#             #         # Use get_resp instead of agent_state.value.invoke
#             #         bot_message = get_resp(full_context)

#             #         chat_history.append((message, f"🤖: {bot_message}"))
#             #     except Exception as e:
#             #         exc_type, exc_value, exc_traceback = sys.exc_info()
#             #         error_details = traceback.extract_tb(exc_traceback)
#             #         error_message = f"An error occurred:\n"
#             #         error_message += f"Type: {exc_type.__name__}\n"
#             #         error_message += f"Message: {str(e)}\n"
#             #         error_message += "Traceback:\n"
#             #         for frame in error_details:
#             #             error_message += f"  File '{frame.filename}', line {frame.lineno}, in {frame.name}\n"
#             #             error_message += f"    {frame.line}\n"
                    
#             #         print(error_message)  # Print to console for debugging
#             #         chat_history.append((message, f"🤖: I'm sorry, but an error occurred. Details:\n{error_message}"))
                
#             #     return "", chat_history, None  # Reset the image input

#             # msg.submit(respond, inputs=[msg, chatbot, image_input], outputs=[msg, chatbot, image_input])
            
#             agent= gr.State()
#             session_id = gr.State()
#             demo.load(get_rag_agent, [], [agent])
    
#             demo.load(lambda: [str(uuid.uuid4())], [], [session_id])
#             # Ensure the chatbot is initialized with the initial prompt
#             demo.load(lambda: [("Hello!", initial_prompt)], [], [chatbot])
#             compile_button.click(compile_c_code, inputs=[code_editor], outputs=[code_output])

#             def respond(message, chat_history, image, agent, session_id, request: gr.Request):
#                 full_context = " ".join([f"User: {msg} AI: {resp}" for msg, resp in chat_history])
#                 full_context += f" User: {message}"
                
#                 # if image is not None:
#                 #     # Only process the image if it's provided
#                 #     image_description = process_image_query(model, tokenizer, image, message)
#                 #     full_context += f" [Image description: {image_description}]"
                    
#                 stdout_handler = StdOutCallbackHandler()
#                 run_id_handler = RunIDCallbackHandler()
                
#                 ret = agent.invoke({"input": full_context}, 
#                                     config={"callbacks": [stdout_handler, run_id_handler]})
                
#                 bot_message = ret["output"]
#                 run_id = run_id_handler.run_id
#                 print(f"run_id: {run_id}")
#                 chat_history.append((message, f"🤖: {bot_message}"))
#                 client_ip = request.client.host
#                 return "", chat_history, None, agent, run_id  # Reset the image input

#             msg.submit(respond, [msg, chatbot, image_input, agent, session_id],
#                         [msg, chatbot, image_input, agent, run_id])
            
#             # Compile function
#             def compile_code(code):
#                 try:
#                     result = compile_c_code(code)
#                     return result
#                 except Exception as e:
#                     return f"Compilation error: {str(e)}"
            
#             compile_button.click(compile_code, inputs=[code_editor], outputs=[code_output])
            
#             # Feedback function
#             def submit_feedback_fn(feedback_text_value, feedback_score_value, run_id_value):
#                 if run_id_value:
#                     client.create_feedback(
#                         run_id=run_id_value,
#                         key="user_feedback",
#                         score=feedback_score_value,
#                         comment=feedback_text_value
#                     )
#                     return "Feedback submitted successfully!"
#                 return "No conversation to provide feedback on."
            
#             submit_feedback.click(
#                 submit_feedback_fn,
#                 inputs=[feedback_text, feedback_score, run_id],
#                 outputs=gr.Textbox(label="Feedback Status")
#             )
            
#             # Logout function
#             def logout():
#                 session_state.value["logged_in"] = False
#                 session_state.value["session_id"] = None
#                 agent_state.value = None  # Clear the agent_state
#                 return gr.update(visible=True), gr.update(visible=False)
            
#             logout_btn.click(logout, outputs=[login_interface, assistant_interface])
        
#         # Control interface visibility
#         def show_interface():
#             if session_state.value["logged_in"]:
#                 return gr.update(visible=False), gr.update(visible=True)
#             else:
#                 return gr.update(visible=True), gr.update(visible=False)
        
#         demo.load(show_interface, outputs=[login_interface, assistant_interface])
    
#     return demo

import gradio as gr
from rag_agent import get_rag_agent, system_message
import logging
import uuid
import os
from jdoodle import compile_c_code
from transformers import AutoModel, AutoTokenizer
import torch
from langsmith import Client
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import base64
import io
from db import authenticate_user, register_user, log_chat, get_chat_history, get_user_by_session

class RunIDCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.run_id = None
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.run_id = kwargs.get("run_id")

# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langsmith-tutorial"

# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY="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"
# LANGCHAIN_PROJECT="pr-extraneous-shopper-77"

API_KEY = 'a3babad21aee482798891f0e56f538f4' #gpt4o
ENDPOINT = 'https://invite-instance-openai.openai.azure.com/'
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY

client = Client()

def get_initial_prompt():
    return """🤖: Hello!
I am an AI Teaching Assistant built to answer all your questions on ECE120.
- You can ask about ECE120, any assignment, and even general concepts such as assembly.
- You can ask about specific concepts such as "What is a logic gate?"
- Seek clarification on course material: "Can you explain the concept of state diagrams with an example?"
- Inquire about assignment instructions: "What should I do if my program doesn't compile?"
- Remember, the more specific your question, the better I can assist you!"""

# Load the MiniCPM-Llama3-V-2_5 model

# model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, device_map='cpu')
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)

def build_app():
    try:
        # Session and agent states
        session_state = gr.State({"logged_in": False, "session_id": None})
        agent_state = gr.State(None)
        run_id = gr.State(None)
        
        with gr.Blocks() as demo:
            gr.HTML("""<h1 align="center">🤖: ECE120 AI-TA</h1>""")
            
            login_interface = gr.Group(visible=lambda: not session_state.value["logged_in"])
            assistant_interface = gr.Group(visible=lambda: session_state.value["logged_in"])
            
            # Login Interface
            with gr.Group(visible=lambda: not session_state.value["logged_in"]) as login_interface:
                gr.Markdown("## Please Log In or Register")
                
                username = gr.Textbox(label="Username", placeholder="Enter your username")
                password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                login_btn = gr.Button("Login")
                register_btn = gr.Button("Register")
                login_output = gr.Textbox(label="Status", interactive=False)
                
                # Login function
                def login(username_input, password_input):
                    session_id_value = authenticate_user(username_input, password_input)
                    if session_id_value:
                        session_state.value["logged_in"] = True
                        session_state.value["session_id"] = session_id_value
                        # Initialize the agent when the user logs in
                        agent_state.value = get_rag_agent()
                        return "Login successful!", gr.update(visible=False), gr.update(visible=True)
                    else:
                        return "Login failed. Please try again.", gr.update(), gr.update()
                
                # Register function
                def register(username_input, password_input):
                    message = register_user(username_input, password_input)
                    return message
                
                login_btn.click(
                    login,
                    inputs=[username, password],
                    outputs=[login_output, login_interface, assistant_interface]
                )
                register_btn.click(register, inputs=[username, password], outputs=[login_output])
                
            with assistant_interface:
                with gr.Row():
                    with gr.Column(scale=3):
                        initial_prompt = get_initial_prompt()
                        chatbot = gr.Chatbot(value=[("Hello!", initial_prompt)], height=600, min_width=300, bubble_full_width=False)
                        with gr.Row():
                            msg = gr.Textbox(placeholder="Insert question here...", scale=4)
                            image_input = gr.Image(type="pil", label="Upload Image (optional)", scale=1)
                            
                        # Define the starter code
                        starter_code = """
    #include <stdio.h>
    int main()
    {
        printf("Hello World");
        return 0;
    }"""
                        # Initialize the code editor with the starter code
                        code_editor = gr.TextArea(value=starter_code, label="C Code Editor", lines=10)
                        compile_button = gr.Button("Compile")
                        code_output = gr.TextArea(placeholder="Output will appear here...", label="Compiler Output", lines=10)
                        clear = gr.ClearButton([msg, chatbot, code_editor, code_output])
                        with gr.Row():
                            feedback_text = gr.Textbox(label="Feedback (optional)")
                            feedback_score = gr.Slider(minimum=0, maximum=1, step=0.1, label="Score")
                            submit_feedback = gr.Button("Submit Feedback")
                            
                        # Add logout button
                        logout_btn = gr.Button("Logout")
                        
                # Define the submit_feedback function
                def submit_feedback_fn(feedback_text_value, feedback_score_value, run_id_value):
                    print(f"Debug - Feedback function received run_id: {run_id_value}")
                    if run_id_value:
                        client.create_feedback(
                            run_id=run_id_value,
                            key="user_feedback",
                            score=feedback_score_value,
                            comment=feedback_text_value
                        )
                        return "Feedback submitted successfully!"
                    return "No conversation to provide feedback on."
                submit_feedback.click(
                    submit_feedback_fn,
                    inputs=[feedback_text, feedback_score, run_id],
                    outputs=gr.Textbox(label="Feedback Status")
                )
                
                def respond(message, chat_history, image, request: gr.Request):
                    if not session_state.value["logged_in"]:
                        return "Please log in to continue.", chat_history, None, None
                    
                    full_context = " ".join([f"User: {msg} AI: {resp}" for msg, resp in chat_history])
                    full_context += f" User: {message}"
                    
                    llm = AzureChatOpenAI(
                        azure_deployment="gpt4-o",  # or your deployment
                        api_version="2024-02-15-preview",  # or your api version
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        # other params...
                    )
                    
                    if image is not None:
                        # Convert the PIL image to base64
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
                        # Prepare the message with both text and image
                        human_message = HumanMessage(
                            content=[
                                {"type": "text", "text": full_context},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                                },
                            ],
                        )
                        
                        # Get response from Azure OpenAI
                        response = llm.invoke([human_message, system_message])
                        bot_message = response.content
                    else:
                        # Use the existing agent for text-only queries
                        if agent_state.value is None:
                            agent_state.value = get_rag_agent()
                        
                        stdout_handler = StdOutCallbackHandler()
                        run_id_handler = RunIDCallbackHandler()
                        ret = agent_state.value.invoke(
                            {
                                "input": full_context,
                            },
                            config={
                                "callbacks": [stdout_handler, run_id_handler]
                            }
                        )
                        bot_message = ret["output"]
                        run_id.value = run_id_handler.run_id
                    
                    print(f"run_id: {run_id.value}")
                    
                    chat_history.append((message, f"🤖: {bot_message}"))
                    client_ip = request.client.host
                    
                    # Log the message and response
                    user_id = get_user_by_session(session_state.value["session_id"])
                    log_chat(user_id, session_state.value["session_id"], message, bot_message)
                    return "", chat_history, None, None  # Reset the image input
                
                # Define the respond function
    #             def respond(message, chat_history, image, request: gr.Request):
    #                 if not session_state.value["logged_in"]:
    #                     return "Please log in to continue.", chat_history, None, None
    #                 if agent_state.value is None:
    #                     agent_state.value = get_rag_agent()
    #                 full_context = " ".join([f"User: {msg} AI: {resp}" for msg, resp in chat_history])
    #                 full_context += f" User: {message}"
                    
    #                 if image is not None:
    #                     # Only process the image if it's provided
    #                     full_context += f" [Image description: {}"
                    
    #                 stdout_handler = StdOutCallbackHandler()
    #                 run_id_handler = RunIDCallbackHandler()
    #                 ret = agent_state.value.invoke(
    #     {
    #         "input": full_context,
    #     },
    #     config={
    #         "callbacks": [stdout_handler, run_id_handler]
    #     }
    # )
    #                 bot_message = ret["output"]
    #                 run_id.value = run_id_handler.run_id
                    
    #                 print(f"run_id: {run_id.value}")
                    
    #                 chat_history.append((message, f"🤖: {bot_message}"))
    #                 client_ip = request.client.host
                    
    #                 # Log the message and response
    #                 user_id = get_user_by_session(session_state.value["session_id"])
    #                 log_chat(user_id, session_state.value["session_id"], message, bot_message)
    #                 return "", chat_history, None, None  # Reset the image input
                
                msg.submit(respond, inputs=[msg, chatbot, image_input], outputs=[msg, chatbot, image_input])
                compile_button.click(compile_c_code, inputs=[code_editor], outputs=[code_output])

                # Logout function
                def logout():
                    session_state.value["logged_in"] = False
                    session_state.value["session_id"] = None
                    agent_state.value = None  # Clear the agent_state
                    return gr.update(visible=True), gr.update(visible=False)
                logout_btn.click(logout, outputs=[login_interface, assistant_interface])
            # Control interface visibility
            def show_interface():
                if session_state.value["logged_in"]:
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=True), gr.update(visible=False)
            demo.load(show_interface, outputs=[login_interface, assistant_interface])
    except Exception as e:
        raise gr.Error(f"Error processing data: {e}")
    
    return demo