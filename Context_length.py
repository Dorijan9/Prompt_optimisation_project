import time
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Hugging Face API token
os.environ["HUGGING_FACE_TOKEN"] = "your-api-key"

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"  # Replace with desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the document content
documents = [
    "The aim of the project is to survey various automated prompt optimisation approaches and conduct a comparative analysis with respect to their applicability on code generation tasks that are performed by LLMs. It seeks to assess the effectiveness of approaches such as Reinforcement Learning (RL), Genetic Algorithms (GA) and Bayesian Optimisation in improving LLM-produced code both with respect to quality and speed, sans manual prompt engineering."
]

# Define the query
query = "What is the main idea of the document?"

# Define context lengths to test
context_lengths = [512, 1024, 2048]

# Prepare results storage
results = []

# Function to generate response
def generate_response(model, tokenizer, text, max_length):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)  # Increase max_new_tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Loop over different context lengths
for length in context_lengths:
    print(f"Testing with context length: {length}")
    # Start timing
    start_time = time.time()
    # Generate response
    response = generate_response(model, tokenizer, query, max_length=length)
    # End timing
    end_time = time.time()
    response_time = end_time - start_time
    
    # Store result
    results.append({
        "context_length": length,
        "response_time": response_time,
        "response": response  # Full response
    })
    
    # Print result for each length
    print(f"Context Length: {length}")
    print(f"Response Time: {response_time:.2f} seconds")
    print(f"Response: {response}\n")

# Display results in a structured format
results_df = pd.DataFrame(results)
print("Summary of Results:")
print(results_df)
