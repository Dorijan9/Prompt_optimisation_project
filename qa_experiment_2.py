import openai
import pandas as pd
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = "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"

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

# Define modified prompting techniques
def standard_prompt(question):
    return f"{question}"

def cot_prompt(question):
    return f"Think step-by-step. {question}"

def get_openai_response(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides clear, concise, and correct answers. Always respond to every question."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=100  # Ensure model generates full responses
    )

    print("RAW RESPONSE:", response)  # Debugging output

    return response["choices"][0]["message"]["content"].strip() if response["choices"] else "No response"

# Function to evaluate if the response matches expected answer
def evaluate_response_exact(expected, response):
    return expected.strip().lower() == response.strip().lower()

# Initialize the results list
results = []

# Loop through the DataFrame rows for each question and prompting method
for idx, row in data.iterrows():
    for prompting_method in [standard_prompt, cot_prompt]:
        prompt = prompting_method(row['Question'])
        response = get_openai_response(prompt, model="gpt-3.5-turbo")  # Using 3.5 model
        
        # Evaluate correctness based on expected answer
        is_correct = evaluate_response_exact(row['Expected Answer'], response)
        results.append({
            "Model": "gpt-3.5-turbo",
            "Category": row["Category"],
            "Question": row["Question"],
            "Expected Answer": row["Expected Answer"],
            "Prompting Method": prompting_method.__name__,
            "Response": response,
            "Accuracy": is_correct
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV for further analysis
results_df.to_csv("model_evaluation_results.csv", index=False)

# Display the DataFrame for quick inspection
print(results_df[['Model', 'Category', 'Prompting Method', 'Expected Answer', 'Response', 'Accuracy']])
