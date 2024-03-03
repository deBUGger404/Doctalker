import gradio as gr
from typing import Any
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

import re
import uuid 
import chromadb


title = 'DocTalker'
description = """ PDFs Conversing Naturally. Extract insights, ask questions, get instant context. Transforming documents into dynamic dialogues. Engage intelligently with your content."""

# enable_box = gr.Textbox.update(value = None, placeholder = 'Upload your OpenAI API key',interactive = True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set', interactive = False)

def set_apikey(api_key: str):
        app.OPENAI_API_KEY = api_key   
        return disable_box

# def enable_api_box():
#         return enable_box

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY: str = None) -> None:
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0
        self.model_type: str = ''
        self.max_token: int = 0

    def __call__(self, file: str) -> Any:
        if self.count==0:
            self.chain = self.build_chain(file)
            self.count+=1
        else:
            self.chain = None
            self.chat_history = []  # Clear chat history
            self.N = 0
            self.count = 0
            self.model_type = ''
            self.max_token = 0
            self.chain = self.build_chain(file)
        return self.chain
    
    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collection
        collection = client.get_or_create_collection(name="my-collection")
        return client
    
    def process_file(self,file: str):
        loader = PyPDFLoader(file.name)
        documents = loader.load()  
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        return documents, file_name
    
    def build_chain(self, file: str):
        if not self.model_type:
            raise ValueError("Model type must be set before building the chain.")
        documents, file_name = self.process_file(file)
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        # print(self.model_type)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(
                        temperature = 0.0, 
                        openai_api_key = self.OPENAI_API_KEY, 
                        model_name = self.model_type,
                        max_tokens = self.max_token
                    ), 
                retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,)
        return chain
    

def get_response(history, query, file): 
        print(app.model_type)
        chain = app.chain
        result = chain({"question": query, 'chat_history':app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]
        app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char
           yield history,''

def drop_value(file, typeM: str):
    app.model_type = typeM
    print(typeM)

def slide_value(file, value: int):
    app.max_token = value  # Assuming you have a max_token attribute in your app class
    print(value)

def confirm_update(file):
    # Call app(file) once after updating attributes
    app(file)
    print('file loaded')
    return [gr.update(visible=True),gr.update(visible=False),gr.update(visible=False)]
     
def update_textbox(selected_value):
    return selected_value
    input_box.value = selected_value

app = my_app()
with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f'<center>{description}</center>')
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(placeholder='Enter OpenAI API key and press enter', show_label=False, interactive=True).style(container=False)
            with gr.Column(scale=0.2):
                # change_api_key = gr.Button('Get OpenAI API Key', link="https://platform.openai.com/account/api-keys")
                gr.Markdown(
                    '<button style="color: black;padding: 7px 20px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;cursor: pointer;border: 2px solid #e5e7ed;border-radius: 8px;background-color: #e9ebee;"><a style="text-decoration: none;color: black;" href="https://platform.openai.com/account/api-keys">Get OpenAI API key here</a></button>'
                )
        with gr.Row():           
            with gr.Column(min_width=400):
                pdf_url = gr.Textbox(label='Enter PDF URL here')
                gr.Markdown("<center><h4>OR<h4></center>")
                btn = gr.File(
                    label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf']
                )
                # btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()
                drop = gr.Dropdown(label="Select Model", info="GEN AI model", choices=["gpt-4-32k", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"])
                slide = gr.Slider(0, 2048, step=20, label="Max Token", info="Change output Token size according",interactive=True)
                update_btn = gr.Button('Update App')
                confirm_btn = gr.Button("Confirm Update", variant="stop", visible=False)
                cancel_btn = gr.Button("Cancel", visible=False)
                drop_qus = gr.Dropdown(
                    label="PRE-SET QUESTIONS", 
                    info="Click a question to automatically populate the input box, and then hit Enter!", 
                    choices=[
                        "Highlighted document equations or formulas?",
                        "What specific areas does this document cover?",
                        "Which theories are explored within this document?",
                        "Are there any notable equations or formulas highlighted in the content?",
                        "What is the central focus of this document or PDF?",
                        "Could you offer a concise overview of the main points in this material?",
                        "What limitations or shortcomings were identified in the study?",
                        "Can you summarize the main contributions or advancements outlined?",
                        "What conclusion or final remarks are provided in this paper?"
                             ]
                    )
            with gr.Column(min_width=600):
                chatbot = gr.Chatbot(value=[], label='Doctalker', elem_id='chatbot').style(height=550)
                input_box = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="Enter text and press enter",
                        )#.style(container=False)
    # with gr.Row():
    #     with gr.Column(min_width=400):
    #         dataset_selection = gr.Dataset(
    #             components=[gr.Textbox(visible=False)],
    #                 label="PRE-SET QUESTIONS: Click a question to automatically populate the input box, and then hit Enter!",
    #                 samples=[
    #                     ["Highlighted document equations or formulas?"],
    #                     ["What specific areas does this document cover?"],
    #                     ["Which theories are explored within this document?"],
    #                     ["Are there any notable equations or formulas highlighted in the content?"],
    #                     ["What is the central focus of this document or PDF?"],
    #                     ["Could you offer a concise overview of the main points in this material?"],
    #                     ["What limitations or shortcomings were identified in the study?"],
    #                     ["Can you summarize the main contributions or advancements outlined?"],
    #                     ["What conclusion or final remarks are provided in this paper?"]
    #                 ],
    #             )
        
        # with gr.Column(min_width=200):
        #     submit_btn = gr.Button('submit')

    print(api_key)
    api_key.submit(
            fn=set_apikey, 
            inputs=[api_key], 
            outputs=[api_key,])

    drop.change(
        fn=drop_value,
        inputs=[btn, drop],
        outputs=[])

    slide.change(
        fn=slide_value,
        inputs=[btn, slide],
        outputs=[])
    
    update_btn.click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [update_btn, confirm_btn, cancel_btn])
    cancel_btn.click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [update_btn, confirm_btn, cancel_btn])
    
    confirm_btn.click(
        fn=confirm_update,
        inputs=[btn],
        outputs=[update_btn, confirm_btn, cancel_btn]
    )

    drop_qus.change(
        fn=update_textbox,
        inputs=[drop_qus],
        outputs=[input_box]
    )

    input_box.submit(
        fn=add_text,
        inputs=[chatbot,input_box],
        outputs=[chatbot, ], 
        queue=False).success(
        fn=get_response,
        inputs = [chatbot, input_box, btn],
        outputs = [chatbot,input_box])

    # submit_btn.click(
    #         fn=add_text, 
    #         inputs=[chatbot,txt], 
    #         outputs=[chatbot, ], 
    #         queue=False).success(
    #         fn=get_response,
    #         inputs = [chatbot, txt, btn],
    #         outputs = [chatbot,txt])

    
demo.queue()
demo.launch() 