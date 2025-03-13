import openai
import csv
import os

# Set OpenAI API Key
openai.api_key = "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"  # Replace with your actual key

# Define different parameter combinations
param_combinations = [
    {"temperature": 1.0, "top_p": 1.0, "label": "High Temp & High Top_p"},
    {"temperature": 0.1, "top_p": 0.1, "label": "Low Temp & Low Top_p"},
    {"temperature": 1.0, "top_p": 0.1, "label": "High Temp & Low Top_p"},
    {"temperature": 0.1, "top_p": 1.0, "label": "Low Temp & High Top_p"},
]

# The question we want to ask
prompt = "Tell me something about space."

# List to store results
results = []

# Loop through parameter combinations and generate responses
for params in param_combinations:
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=200
    )

    generated_text = response["choices"][0]["message"]["content"]
    
    # Store the results in a dictionary
    results.append({
        "Parameter Setting": params["label"],
        "Temperature": params["temperature"],
        "Top_p": params["top_p"],
        "Generated Response": generated_text
    })

# Save results to a CSV file
csv_filename = "openai_responses.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Parameter Setting", "Temperature", "Top_p", "Generated Response"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_filename}")
