# Import required libraries
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b"
headers = {"Authorization": "Bearer your-api-key"}

# Define modified prompting techniques
def standard_prompt(question):
    return f"{question}"

def cot_prompt(question):
    return f"Think step-by-step. {question}"

# Function to get the model response using Hugging Face Inference API with retries
def get_model_response(api_url, headers, prompt, max_retries=10, wait_time=20):
    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            response_json = response.json()
            return response_json[0]["generated_text"] if response_json else ""
        elif response.status_code == 503:
            print(f"Error: {response.status_code}, {response.json()['error']}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return ""
    return ""  # Return an empty response if all retries fail

# Set up question-answer pairs with simplified instructions
data = pd.DataFrame({
    'Category': ['Math', 'Reasoning', 'Coding', 'Multiple-Choice'],
    'Question': [
        'What is 2 + 2? Answer with a number.', 
        'If today is Monday, what day is tomorrow? Answer with one word.', 
        'Write a Python function to add two numbers. Answer with code only.', 
        'What is the capital of France? Choose A, B, or C.\nA Berlin B Paris C Rome'
    ],
    'Expected Answer': ['4', 'Tuesday', 'def add(a, b): return a+b', 'B']
})

# Function to evaluate if the response exactly matches the expected answer with no additional text
def evaluate_response_exact(expected, response):
    return expected.strip().lower() == response.strip().lower()

# Initialize the results list
results = []

# Loop through the DataFrame rows for each question and prompting method
for idx, row in data.iterrows():
    for prompting_method in [standard_prompt, cot_prompt]:
        prompt = prompting_method(row['Question'])
        response = get_model_response(API_URL, headers, prompt)  # Use API with retry mechanism
        
        # Evaluate correctness based on expected answer
        is_correct = evaluate_response_exact(row['Expected Answer'], response)
        results.append({
            "Model": "LLaMA-2-7B",
            "Category": row["Category"],
            "Question": row["Question"],
            "Expected Answer": row["Expected Answer"],
            "Prompting Method": prompting_method.__name__,
            "Response": response,
            "Accuracy": is_correct
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Visualize Results
# Group by Model, Prompting Method, and Category and calculate mean accuracy
accuracy = results_df.groupby(['Model', 'Prompting Method', 'Category'])['Accuracy'].mean().unstack()
accuracy.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance by Prompting Method and Category")
plt.ylabel("Accuracy")
plt.show()

# Save Results to CSV for further analysis
results_df.to_csv("model_evaluation_results.csv", index=False)

# Display the DataFrame for quick inspection
print(results_df[['Model', 'Category', 'Prompting Method', 'Expected Answer', 'Response', 'Accuracy']])
