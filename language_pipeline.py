from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_huggingface import HuggingFacePipeline

# Step 1: Initialize the Hugging Face text generation pipeline
generator = pipeline("text-generation", model="distilgpt2")

# Step 2: Wrap the Hugging Face model pipeline in LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Step 3: Set up a LangChain prompt template
template = "Explain the impact of AI in job markets: {input_text}"
prompt = PromptTemplate(template=template, input_variables=["input_text"])

# Step 4: Create an LLMChain with the prompt and model
chain = LLMChain(llm=llm, prompt=prompt)

# Step 5: Define an input for the prompt and run the chain
input_text = "What is the future of work with AI?"
response = chain.run(input_text=input_text)

# Step 6: Output the result
print("Generated Response:", response)
