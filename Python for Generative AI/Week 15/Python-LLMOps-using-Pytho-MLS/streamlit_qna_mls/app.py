import os
import uuid
import json

import gradio as gr

from openai import OpenAI

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from huggingface_hub import CommitScheduler
from pathlib import Path


client = OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=os.environ['ANYSCALE_API_KEY']
)

embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

streamlit_collection = 'streamlit'

vectorstore_persisted = Chroma(
    collection_name=streamlit_collection,
    persist_directory='./streamlitdb',
    embedding_function=embedding_model
)

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

# Prepare the logging functionality

log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="document-qna-chroma-anyscale-logs",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

qna_system_message = """
You are an assistant to a coder. Your task is to provide relevant information about the Python package Streamlit.

User input will include the necessary context for you to answer their questions. This context will begin with the token: ###Context.
The context contains references to specific portions of documents relevant to the user's query, along with source links.
The source for a context will begin with the token ###Source

When crafting your response:
1. Select the most relevant context or contexts to answer the question.
2. Include the source links in your response.
3. User questions will begin with the token: ###Question.
4. If the question is irrelevant to streamlit respond with - "I am an assistant for streamlit Docs. I can only help you with questions related to streamlit"

Please adhere to the following guidelines:
- Answer only using the context provided.
- Do not mention anything about the context in your final answer.
- If the answer is not found in the context, it is very very important for you to respond with "I don't know. Please check the docs @ 'https://docs.streamlit.io/'"
- Always quote the source when you use the context. Cite the relevant source at the end of your response under the section - Sources:
- Do not make up sources. Use the links provided in the sources section of the context and nothing else. You are prohibited from providing other links/sources.

Here is an example of how to structure your response:

Answer:
[Answer]

Source
[Source]
"""

qna_user_message_template = """
###Context
Here are some documents that are relevant to the question.
{context}
```
{question}
```
"""

# Define the predict function that runs when 'Submit' is clicked or when a API request is made
def predict(user_input):

    relevant_document_chunks = retriever.invoke(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ".".join(context_list)
    
    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model='mistralai/Mixtral-8x7B-Instruct-v0.1',
            messages=prompt,
            temperature=0
        )

        prediction = response.choices[0].message.content

    except Exception as e:
        prediction = e

    # While the prediction is made, log both the inputs and outputs to a local log file
    # While writing to the log file, ensure that the commit scheduler is locked to avoid parallel
    # access
    
    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(json.dumps(
                {
                    'user_input': user_input,
                    'retrieved_context': context_for_query,
                    'model_response': prediction
                }
            ))
            f.write("\n")
    
    return prediction


textbox = gr.Textbox(placeholder="Enter your query here", lines=6)

# Create the interface
demo = gr.Interface(
    inputs=textbox, fn=predict, outputs="text",
    title="Streamlit Q&A System",
    description="This web API presents an interface to ask questions on streamlit documentation",
    article="Note that questions that are not relevant to streamlit or not within the sample documents will be answered with 'I don't know. Please check the docs @ 'https://docs.streamlit.io/''",
    concurrency_limit=16
)

demo.queue()
demo.launch()