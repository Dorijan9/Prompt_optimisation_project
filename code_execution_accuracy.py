import subprocess
import time
import openai  # Replace with an appropriate LLM library (e.g., llama_index, transformers)
import pandas as pd

# Define OpenAI API Key
OPENAI_API_KEY = "api-key"
openai.api_key = OPENAI_API_KEY  # Set API Key globally

# Define base coding challenges
challenges = {
    "python": "Write a Python function to compute the sum of a list of numbers.",
    "javascript": "Write a JavaScript function to compute the sum of an array of numbers.",
    "cpp": "Write a C++ program to compute the sum of numbers in an array."
}

def generate_code_prompt_variations(base_prompt, model="gpt-4"):
    "Use LLM to generate multiple optimized versions of the given coding challenge."
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code evaluator. Generate optimized prompts for coding challenges."},
            {"role": "user", "content": f"Generate 3 variations of this prompt for better code quality and performance:\n{base_prompt}"}
        ]
    )
    return [choice["message"]["content"] for choice in response["choices"]]

def execute_code(language, code):
    "Executes code in the given language and captures output."
    start_time = time.time()
    try:
        if language == "python":
            result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=5)
        elif language == "javascript":
            result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=5)
        elif language == "cpp":
            with open("test.cpp", "w") as f:
                f.write(code)
            subprocess.run(["g++", "test.cpp", "-o", "test.out"], capture_output=True, text=True)
            result = subprocess.run(["./test.out"], capture_output=True, text=True, timeout=5)
        else:
            return "Unsupported language", "", 0.0
        
        execution_time = time.time() - start_time
        return result.stdout, result.stderr, execution_time
    except subprocess.TimeoutExpired:
        return "Timeout", "", 5.0
    except Exception as e:
        return "", str(e), 0.0

def evaluate_prompt_optimization():
    "Runs the optimized prompt tests and evaluates performance."
    results = []
    for lang, base_prompt in challenges.items():
        variations = generate_code_prompt_variations(base_prompt)
        for i, variation in enumerate(variations):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate optimized code based on the given prompt."},
                    {"role": "user", "content": variation}
                ]
            )
            generated_code = response["choices"][0]["message"]["content"]
            output, error, exec_time = execute_code(lang, generated_code)
            
            results.append({
                "Language": lang,
                "Prompt Variation": variation,
                "Generated Code": generated_code,
                "Output": output,
                "Error": error,
                "Execution Time (s)": exec_time
            })
    return results

if __name__ == "__main__":
    results_df = pd.DataFrame(evaluate_prompt_optimization())
    results_df.to_csv("prompt_optimization_results.csv", index=False)
    print("Results saved to prompt_optimization_results.csv")
