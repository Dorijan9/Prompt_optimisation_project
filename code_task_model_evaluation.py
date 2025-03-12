import openai
import os

# Set OpenAI API Key
openai.api_key = "api-key"  

# Define the OpenAI Model
model_name = "gpt-4-turbo"

# Define code-related tasks
tasks = ["Code Completion", "Code Generation", "Debugging", "Code Explanation"]

# Sample prompts for evaluation
prompts = {
    "Code Completion": "def fibonacci(n):\n    # Complete the function\n",
    "Code Generation": "Write a Python function to sort a list using quicksort.",
    "Debugging": "Fix the bug in this Python function:\ndef add_numbers(a, b):\n    return a - b",
    "Code Explanation": "Explain what this Python code does:\ndef factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
}

# Function to generate responses using OpenAI
def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# Function to evaluate model responses
def evaluate_response(task, response):
    correctness, coherence, reasoning_depth = 0, 0, 0

    # Basic correctness checks
    if task == "Code Completion" and "return" in response and "if" in response:
        correctness = 1
    elif task == "Code Generation" and "def" in response and "return" in response:
        correctness = 1
    elif task == "Debugging" and "+" in response:
        correctness = 1
    elif task == "Code Explanation" and ("recursive" in response.lower() or "factorial" in response.lower()):
        correctness = 1

    # Coherence & reasoning depth (word heuristics)
    coherence = min(1, len(response.split()) / 15)  
    reasoning_depth = min(1, len(set(response.split())) / 15)

    return correctness, coherence, reasoning_depth

# Evaluate model on tasks and print results
for task, prompt in prompts.items():
    print(f"\n--- Task: {task} ---")
    response = generate_response(prompt)
    correctness, coherence, reasoning_depth = evaluate_response(task, response)

    print(f"Prompt:\n{prompt}")
    print(f"GPT-4-Turbo Response:\n{response}")
    print(f"Correctness: {correctness}")
    print(f"Coherence: {coherence:.2f}")
    print(f"Reasoning Depth: {reasoning_depth:.2f}")
    print("-" * 50)
