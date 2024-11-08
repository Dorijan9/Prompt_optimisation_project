from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the app
app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "gpt2"  # Change to your preferred model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define a prediction endpoint
@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}

# Run the server with: uvicorn app:app --reload
