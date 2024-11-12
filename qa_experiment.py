# Import required libraries
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Set up question-answer pairs with simplified instructions
data = pd.DataFrame({
    'Category': ['Math', 'Math', 'Reasoning', 'Reasoning', 'Coding', 'Coding', 'Multiple-Choice', 'Multiple-Choice'],
    'Question': [
        'What is 2 + 2? Answer with a number.', 
        'What is 2 + 2? Answer with a number.', 
        'If today is Monday, what day is tomorrow? Answer with one word.', 
        'If today is Monday, what day is tomorrow? Answer with one word.', 
        'Write a Python function to add two numbers. Answer with code only.', 
        'Write a Python function to add two numbers. Answer with code only.',
        'What is the capital of France? Choose A, B, or C.\nA) Berlin B) Paris C) Rome',
        'What is the capital of France? Choose A, B, or C.\nA) Berlin B) Paris C) Rome'
    ],
    'Expected Answer': ['4', '4', 'Tuesday', 'Tuesday', 'def add(a, b): return a+b', 'def add(a, b): return a+b', 'B', 'B']
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

# Define modified prompting techniques
def standard_prompt(question):
    return f"{question}"

def cot_prompt(question):
    return f"Think step-by-step. {question}"

# Function to get the model response with limited response length
def get_model_response(model, tokenizer, prompt):
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Tokenize the input with padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate the response with do_sample enabled
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,  # Limit response length
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.3,  # Lower temperature for less randomness
        top_p=0.9,  # Focused sampling
        do_sample=True  # Enable sampling for temperature and top_p to work
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response by removing prompt text if it appears in the output
    response = response.replace(prompt, "").strip()
    response = response.splitlines()[0]  # Take only the first line of the response
    
    return response

# Run Experiment and Collect Responses (correct loop to avoid extra runs)
results = []

for model_name, (tokenizer, model) in models.items():
    for idx, row in data.iterrows():
        for prompting_method in [standard_prompt, cot_prompt]:
            prompt = prompting_method(row['Question'])
            print(f"Processing: Model={model_name}, Category={row['Category']}, Prompting Method={prompting_method.__name__}, Question='{row['Question']}'")  # Debugging output
            response = get_model_response(model, tokenizer, prompt)
            print(f"Response: {response}")  # Debugging output
            results.append({
                "Model": model_name,
                "Category": row["Category"],
                "Question": row["Question"],
                "Expected Answer": row["Expected Answer"],
                "Prompting Method": prompting_method.__name__,
                "Response": response
            })
            
            # Break after two entries for each category to ensure only two responses per prompt type
            if len(results) >= (idx + 1) * 2:
                break

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Partial Match Evaluation Function
def evaluate_response_partial(expected, response):
    return expected.strip().lower() in response.strip().lower()

# Apply the partial match evaluation function
results_df['Correct'] = results_df.apply(lambda x: evaluate_response_partial(x['Expected Answer'], x['Response']), axis=1)

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
print(results_df[['Model', 'Category', 'Prompting Method', 'Expected Answer', 'Response', 'Correct']])