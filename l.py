git clone https://github.com/afiorg9000/context_data.git

!pip install llama-index
!pip install langchain

from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True: 
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        display(Markdown(f"Response: <b>{response.response}</b>"))
        
        os.environ["OPENAI_API_KEY"] = input("Paste your OpenAI key here and hit enter:")
        
        construct_index("context_data/data")
        
        
        "TypeError                                 Traceback (most recent call last)
<ipython-input-5-d6115cbbba37> in <module>
----> 1 construct_index("context_data/data")

2 frames
/usr/local/lib/python3.9/dist-packages/llama_index/indices/vector_store/base.py in __init__(self, nodes, index_struct, service_context, text_qa_template, vector_store, use_async, **kwargs)
     55         self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
     56         self._use_async = use_async
---> 57         super().__init__(
     58             nodes=nodes,
     59             index_struct=index_struct,

TypeError: __init__() got an unexpected keyword argument 'llm_predictor'"