import os
import uuid
import joblib
import json

import gradio as gr
import pandas as pd

from huggingface_hub import CommitScheduler
from pathlib import Path

# Run the training script placed in the same directory as app.py
# The training script will train and persist a logistic regression
# model with the filename 'model.joblib'

os.system("python train.py")

# Load the freshly trained model from disk

insurance_charge_predictor = joblib.load('model.joblib')

# Prepare the logging functionality

log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="insurance-charge-mlops-logs",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

# Define the predict function that runs when 'Submit' is clicked or when a API request is made
def predict_insurance_charge(age, bmi, children,sex, smoker, region):
    sample = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }
    
    data_point = pd.DataFrame([sample])
    prediction = insurance_charge_predictor.predict(data_point).tolist()

    # While the prediction is made, log both the inputs and outputs to a local log file
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
            
    return round(prediction[0],2)

# Set up UI components for input and output

age_input = gr.Number(label='age')
bmi_input = gr.Number(label='bmi')
children_input = gr.Number(label='children')
sex_input = gr.Dropdown(['female','male'],label='sex')
smoker_input = gr.Dropdown(['yes','no'],label='smoker')
region_input = gr.Dropdown(
    ['southeast', 'southwest', 'northwest', 'northeast'],
    label='region'
)

model_output = gr.Label(label="Insurance Charges")

# Create the interface
demo = gr.Interface(
    fn=predict_insurance_charge,
    inputs=[age_input, bmi_input, children_input,sex_input, smoker_input, region_input],
    outputs=model_output,
    title="HealthyLife Insurance Charge Prediction",
    description="This API allows you to predict the estimating insurance charges based on customer attributes",
    examples=[[33,33.44,5,'male','no','southeast'],
              [58,25.175,0,'male','no','northeast'],
              [52,38.380,2,'female','no','northeast']],
    concurrency_limit=16
)

# Launch with a load balancer
demo.queue()
demo.launch(share=False)
