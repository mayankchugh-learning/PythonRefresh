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

embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-small')

tesla_10k_collection = 'tesla-10k-2019-to-2023'

vectorstore_persisted = Chroma(
    collection_name=tesla_10k_collection,
    persist_directory='./tesla_db',
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
You are an assistant to a financial services firm who answers user queries on annual reports.
Users will ask questions delimited by triple backticks, that is, ```.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.
Please answer only using the context provided in the input. However, do not mention anything about the context in your answer. 
If the answer is not found in the context, respond "I don't know".
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
            model='mlabonne/NeuralHermes-2.5-Mistral-7B',
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
    title="AMA on Tesla 10-K statements",
    description="This web API presents an interface to ask questions on contents of the Tesla 10-K reports for the period 2019 - 2023.",
    article="Note that questions that are not relevant to the Tesla 10-K report will not be answered.",
    examples=[["What was the total revenue of the company in 2022?", "$ 81.46 Billion"],
              ["Summarize the Management Discussion and Analysis section of the 2021 report in 50 words.", ""],
              ["What was the company's debt level in 2020?", ""],
              ["Identify 5 key risks identified in the 2019 10k report? Respond with bullet point summaries.", ""]
             ],
    concurrency_limit=16
)

demo.queue()
demo.launch()