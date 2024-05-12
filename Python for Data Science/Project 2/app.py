# Import the libraries
import os
import uuid
import joblib
import json

import gradio as gr
import pandas as pd

from huggingface_hub import CommitScheduler
from pathlib import Path


# Run the training script placed in the same directory as app.py
# The training script will train and persist a linear regression
# model with the filename 'model.joblib'




# Load the freshly trained model from disk


# Prepare the logging functionality
log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="-----------",  # provide a name "insurance-charge-mlops-logs" for the repo_id
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

# Define the predict function which will take features, convert to dataframe and make predictions using the saved model
# the functions runs when 'Submit' is clicked or when a API request is made
model = joblib.load('model.joblib')

age_input = gr.Number(label="Age")
bmi_input = gr.Number(label="BMI")
children_input = gr.Dropdown(["0", "1", "2", "3", "4", "5"],label='Children')
sex_input = gr.Dropdown(['female','male'],label='Sex')
smoker_input = gr.Dropdown(['yes', 'no'],label='Smoker')
region_input = gr.Dropdown(['southwest' 'southeast' 'northwest' 'northeast'],label='Region')

model_output = gr.Label(label="HealthyLife Insurance Charge Prediction - Predict the cost for your Medical Insurance!")


def predict_insurance_charge(age, bmi, children, sex, smoker, region):
    sample = {
        'Age': age,
        'BMI': bmi,
        'Children': children,
        'Sex': sex,
        'Smoker': smoker,
        'Region': region
    }
    data_point = pd.DataFrame([sample])
    prediction = predict_insurance_charge.predict(data_point).tolist()


    # While the prediction is made, log both the inputs and outputs to a  log file
    # While writing to the log file, ensure that the commit scheduler is locked to avoid parallel
    # access

    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(json.dumps(
                {
                    'age': age,
                    'bmi': bmi,
                    'children': children,
                    'sex': sex,
                    'smoker': smoker,
                    'region': region,
                    'prediction': prediction[0]
                }
            ))
            f.write("\n")

    return prediction[0]



# Set up UI components for input and output
demo = gr.Interface(
    fn=predict_insurance_cost,
    #age, bmi, children, sex, smoker, region
    inputs=[age_input, bmi_input, children_input, 
            sex_input, smoker_input, region_input],
    outputs=model_output,
    title="HealthyLife Insurance Charge Predictor",
    description="This API allows you to predict the HealthyLife Insurance Charge",
    allow_flagging="auto",
    concurrency_limit=16
)


# Create the gradio interface, make title "HealthyLife Insurance Charge Prediction"


# Launch with a load balancer
demo.queue()
demo.launch(share=False)
