# Import required libraries
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

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

# Load Models
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

model = {
    "GPT-Neo": load_model("EleutherAI/gpt-neo-125M"),
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

# Function to evaluate if the response exactly matches the expected answer with no additional text
def evaluate_response_exact(expected, response):
    return expected.strip().lower() == response.strip().lower()

# Initialize the results list
results = []

# Loop through the DataFrame rows for each question and prompting method
for idx, row in data.iterrows():
    for prompting_method in [standard_prompt, cot_prompt]:
        prompt = prompting_method(row['Question'])
        #print(f"Processing: Prompting Method={prompting_method.__name__}, Question='{row['Question']}'")  # Debugging output
        response = get_model_response(model["GPT-Neo"][1], model["GPT-Neo"][0], prompt)  # Use model directly
        #print(f"Response: {response}")  # Debugging output
        
        # Evaluate correctness based on expected answer
        is_correct = evaluate_response_exact(row['Expected Answer'], response)
        results.append({
            "Model": "GPT-Neo",
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
