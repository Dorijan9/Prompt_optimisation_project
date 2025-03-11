import subprocess
import time

# Define coding challenges with expected outputs
challenges = {
    "python": [
        {"code": "print(sum([1, 2, 3, 4, 5]))", "expected": "15\n"},
        {"code": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)\nprint(factorial(5))", "expected": "120\n"},
    ],
    "javascript": [
        {"code": "console.log([1,2,3,4,5].reduce((a, b) => a + b, 0));", "expected": "15\n"},
        {"code": "function factorial(n) { return n === 0 ? 1 : n * factorial(n-1); }\nconsole.log(factorial(5));", "expected": "120\n"},
    ],
    "cpp": [
        {"code": "#include <iostream>\nusing namespace std;\nint main() { cout << 1+2+3+4+5 << endl; return 0; }", "expected": "15\n"},
        {"code": "#include <iostream>\nusing namespace std;\nint factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }\nint main() { cout << factorial(5) << endl; return 0; }", "expected": "120\n"},
    ],
}

# Execution function
def execute_code(language, code):
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
            return "Unsupported language"
        
        execution_time = time.time() - start_time
        return result.stdout, result.stderr, execution_time
    except subprocess.TimeoutExpired:
        return "Timeout", "", 5.0
    except Exception as e:
        return "", str(e), 0.0

# Run tests and evaluate accuracy
def evaluate_accuracy():
    results = []
    for lang, tests in challenges.items():
        for test in tests:
            output, error, exec_time = execute_code(lang, test["code"])
            correct = (output == test["expected"]) and not error
            results.append({
                "Language": lang,
                "Code": test["code"],
                "Expected": test["expected"],
                "Output": output,
                "Error": error,
                "Execution Time (s)": exec_time,
                "Correct": correct
            })
    return results

# Run evaluation and print results
if __name__ == "__main__":
    import pandas as pd
    results_df = pd.DataFrame(evaluate_accuracy())
    results_df.to_csv("code_execution_results.csv", index=False)
    print("Results saved to code_execution_results.csv")