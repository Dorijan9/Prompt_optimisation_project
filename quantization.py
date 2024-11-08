# This script demonstrates loading an open-source model, applying quantization, 
# and comparing inference results and model size.

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.quantization import quantize_dynamic

# Set the quantization backend to 'qnnpack'
torch.backends.quantized.engine = 'qnnpack'

# Load a pretrained model and tokenizer
model_name = "distilbert-base-uncased"  # Example model, replace with the desired model name
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply dynamic quantization
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Test with sample text
text = "I love using open-source models for natural language processing!"
inputs = tokenizer(text, return_tensors="pt")

# Run inference
with torch.no_grad():
    original_output = model(**inputs).logits
    quantized_output = quantized_model(**inputs).logits

print("Original Model Output:", original_output)
print("Quantized Model Output:", quantized_output)

# Optional: Calculate the difference between original and quantized outputs
difference = torch.abs(original_output - quantized_output).mean().item()
print(f"Mean absolute difference between original and quantized model outputs: {difference:.4f}")

# Save the Quantized Model (Optional)
# quantized_model.save_pretrained("quantized_model")
