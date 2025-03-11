# Filename: code_task_model_evaluation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Define models
models = {
    "Llama 3": "meta-llama/Meta-Llama-3-8B",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "Falcon": "tiiuae/falcon-7b",
    "GPT-NeoX": "EleutherAI/gpt-neox-20b"
}

# Define code-related tasks
tasks = ["Code Completion", "Code Generation", "Debugging", "Code Explanation"]

# Sample prompts for evaluation
prompts = {
    "Code Completion": "def fibonacci(n):\n    # Complete the function\n",
    "Code Generation": "Write a Python function to sort a list using quicksort.",
    "Debugging": "Fix the bug in this Python function:\ndef add_numbers(a, b):\n    return a - b",
    "Code Explanation": "Explain what this Python code does:\ndef factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
}

# Score categories
scores = {model: {task: {"Correctness": 0, "Coherence": 0, "Reasoning Depth": 0} for task in tasks} for model in models}

# Function to evaluate model responses
def evaluate_response(model_name, task, response):
    """
    Assigns scores for correctness, coherence, and reasoning depth based on heuristic rules.
    """
    correctness, coherence, reasoning_depth = 0, 0, 0

    # Basic correctness checks
    if task == "Code Completion":
        if "return" in response and "if" in response:
            correctness = 1
    elif task == "Code Generation":
        if "def" in response and "return" in response:
            correctness = 1
    elif task == "Debugging":
        if "+" in response:  # Checking if the error was fixed (addition instead of subtraction)
            correctness = 1
    elif task == "Code Explanation":
        if "recursive" in response.lower() or "factorial" in response.lower():
            correctness = 1

    # Coherence (length and structure of response)
    coherence = min(1, len(response.split()) / 15)  # Longer responses tend to be better

    # Reasoning depth (simple heuristic based on vocabulary richness)
    reasoning_depth = min(1, len(set(response.split())) / 15)

    return correctness, coherence, reasoning_depth

# Load models and evaluate
for model_name, model_path in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)

    for task, prompt in prompts.items():
        response = text_pipeline(prompt)[0]["generated_text"]
        correctness, coherence, reasoning_depth = evaluate_response(model_name, task, response)

        scores[model_name][task]["Correctness"] = correctness
        scores[model_name][task]["Coherence"] = coherence
        scores[model_name][task]["Reasoning Depth"] = reasoning_depth

# Convert results to a plottable format
import pandas as pd
data = []
for model, task_scores in scores.items():
    for task, score_dict in task_scores.items():
        for score_type, value in score_dict.items():
            data.append([model, task, score_type, value])

df = pd.DataFrame(data, columns=["Model", "Task", "Score Type", "Score"])

# Plot performance
plt.figure(figsize=(12, 6))
sns.barplot(x="Task", y="Score", hue="Model", data=df[df["Score Type"] == "Correctness"])
plt.title("Correctness Score Comparison (Code Tasks)")
plt.legend(title="Models")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Task", y="Score", hue="Model", data=df[df["Score Type"] == "Coherence"])
plt.title("Coherence Score Comparison (Code Tasks)")
plt.legend(title="Models")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Task", y="Score", hue="Model", data=df[df["Score Type"] == "Reasoning Depth"])
plt.title("Reasoning Depth Score Comparison (Code Tasks)")
plt.legend(title="Models")
plt.show()
