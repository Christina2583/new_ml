import mlflow
import mlflow.pytorch
import torch
from transformers import AutoModel, AutoTokenizer

mlflow.set_experiment("DeepSeek_Experiments")


model_name = "deepseek-ai/deepseek-coder-6.7b"
model ="deepseek-ai"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", model_name)
    
    # Log some dummy metrics
    mlflow.log_metric("accuracy", 0.85)
    
    # Save model
    mlflow.pytorch.log_model(model, "deepseek_model")
    
    print("Experiment logged successfully!")
