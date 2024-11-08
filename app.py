from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define request model
class TextRequest(BaseModel):
    text: str

# Initialize the app
app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define a prediction endpoint
@app.post("/predict")
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    
    # Adjust generation parameters for diversity and reduce repetition
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=50, 
        temperature=1.2,   # Lower temperature for controlled randomness
        top_k=100,          # Limit to top 50 words for sampling
        top_p=0.8,         # Use nucleus sampling with 90% cumulative probability
        repetition_penalty=1.5  # Penalty for repeating the same phrase
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}

# Run the server with: uvicorn app:app --reload
