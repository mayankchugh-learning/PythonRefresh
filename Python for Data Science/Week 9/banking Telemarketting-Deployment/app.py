import os
import uuid
import joblib
import json

import gradio as gr
import pandas as pd

from huggingface_hub import CommitScheduler
from pathlib import Path

log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="term-deposit-logs",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

term_deposit_predictor = joblib.load('model.joblib')

age_input = gr.Number(label="Age")
duration_input = gr.Number(label='Duration(Sec)')
cc_contact_freq_input = gr.Number(label='CC Contact Freq')
days_since_pc_input = gr.Number(label='Days Since PC')
pc_contact_freq_input = gr.Number(label='Pc Contact Freq')
job_input = gr.Dropdown(['admin.', 'blue-collar', 'technician', 'services', 'management',
       'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed',
       'student', 'unknown'],label="Job")
marital_input = gr.Dropdown(['married', 'single', 'divorced', 'unknown'],label='Marital Status')
education_input = gr.Dropdown(['experience', 'university degree', 'high school', 'professional.course',
       'Others', 'illiterate'],label='Education')
defaulter_input = gr.Dropdown(['no', 'unknown', 'yes'],label='Defaulter')
home_loan_input = gr.Dropdown(['yes', 'no', 'unknown'],label='Home Loan')
personal_loan_input = gr.Dropdown(['yes', 'no', 'unknown'],label='Personal Loan')
communication_type_input = gr.Dropdown(['cellular', 'telephone'],label='Communication Type')
last_contacted_input = gr.Dropdown(['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'mar', 'sep', 'dec'],label='Last Contacted')
day_of_week_input = gr.Dropdown(['thu', 'mon', 'wed', 'tue', 'fri'],label='Day of Week')
pc_outcome_input = gr.Dropdown(['nonexistent', 'failure', 'success'], label='PC Outcome')


model_output = gr.Label(label="Subscribed")

def predict_term_deposit(age, duration, cc_contact_freq, days_since_pc, pc_contact_freq, job, marital_status, education, 
                         defaulter, home_loan, personal_loan, communication_type, last_contacted, 
                         day_of_week, pc_outcome):
    sample = {
        'Age': age,
        'Duration(Sec)': duration,
        'CC Contact Freq': cc_contact_freq,
        'Days Since PC': days_since_pc,
        'PC Contact Freq': pc_contact_freq,
        'Job': job,
        'Marital Status': marital_status,
        'Education': education,
        'Defaulter': defaulter,
        'Home Loan': home_loan,
        'Personal Loan': personal_loan,
        'Communication Type': communication_type,
        'Last Contacted': last_contacted,
        'Day of Week': day_of_week,
        'PC Outcome': pc_outcome,
    }
    data_point = pd.DataFrame([sample])
    prediction = term_deposit_predictor.predict(data_point).tolist()

    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(json.dumps(
                {
                    'Age': age,
                    'Duration(Sec)': duration,
                    'CC Contact Freq': cc_contact_freq,
                    'Days Since PC': days_since_pc,
                    'PC Contact Freq': pc_contact_freq,
                    'Job': job,
                    'Marital Status': marital_status,
                    'Education': education,
                    'Defaulter': defaulter,
                    'Home Loan': home_loan,
                    'Personal Loan': personal_loan,
                    'Communication Type': communication_type,
                    'Last Month Contacted': last_contacted,
                    'Day of Week': day_of_week,
                    'PC Outcome': pc_outcome,
                    'prediction': prediction[0]
                }
            ))
            f.write("\n")
            
    return prediction[0]

demo = gr.Interface(
    fn=predict_term_deposit,
    inputs=[age_input,
            duration_input,
            cc_contact_freq_input,
            days_since_pc_input,
            pc_contact_freq_input,
            job_input,
            marital_input,
            education_input,
            defaulter_input,
            home_loan_input,
            personal_loan_input,
            communication_type_input,
            last_contacted_input,
            day_of_week_input,
            pc_outcome_input],
    outputs=model_output,
    title="Term Deposit Prediction",
    description="This API allows you to predict the person who are going to likely subscribe the term deposit",
    allow_flagging="auto",
    concurrency_limit=8
)

demo.queue()
demo.launch(share=False)