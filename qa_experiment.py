#/usr/bin/python3 -m pip install
# Import required libraries
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Set up question-answer pairs
data = pd.DataFrame({
    'Category': ['Math', 'Reasoning', 'Coding', 'Multiple-Choice'],
    'Question': [
        'What is 2 + 2?', 
        'If today is Monday, what day is tomorrow?', 
        'Write a Python function to add two numbers.', 
        'What is the capital of France? A) Berlin B) Paris C) Rome'
    ],
    'Expected Answer': ['4', 'Tuesday', 'def add(a, b): return a+b', 'B']
})

# Load Models
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

models = {
    "GPT-Neo": load_model("EleutherAI/gpt-neo-125M"),
    # Add other models here as needed
}

# Define Prompting Techniques
def standard_prompt(question):
    return question

def cot_prompt(question):
    return f"Let's think through this step-by-step. {question}"

# Function to Get Model Response
def get_model_response(model, tokenizer, prompt):
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Tokenize the input with padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate the response with explicit pad_token_id and attention_mask
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Run Experiment and Collect Responses
results = []

for model_name, (tokenizer, model) in models.items():
    for idx, row in data.iterrows():
        for prompting_method in [standard_prompt, cot_prompt]:
            prompt = prompting_method(row['Question'])
            response = get_model_response(model, tokenizer, prompt)
            results.append({
                "Model": model_name,
                "Category": row["Category"],
                "Question": row["Question"],
                "Expected Answer": row["Expected Answer"],
                "Prompting Method": prompting_method.__name__,
                "Response": response
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Evaluation Function for Case-Insensitive Exact Match (Option 1)
def evaluate_response(expected, response):
    return expected.strip().lower() == response.strip().lower()

# Apply the evaluation function
results_df['Correct'] = results_df.apply(lambda x: evaluate_response(x['Expected Answer'], x['Response']), axis=1)

# Visualize Results
# Group by Model, Prompting Method, and Category and calculate mean accuracy
accuracy = results_df.groupby(['Model', 'Prompting Method', 'Category'])['Correct'].mean().unstack()
accuracy.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance by Prompting Method and Category")
plt.ylabel("Accuracy")
plt.show()

# Save Results to CSV for further analysis
results_df.to_csv("model_evaluation_results.csv", index=False)

# Display the DataFrame for quick inspection
print(results_df[['Model', 'Category', 'Prompting Method', 'Expected Answer', 'Response', 'Correct']].head())
