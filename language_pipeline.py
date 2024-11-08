from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_huggingface import HuggingFacePipeline

#Initialize the Hugging Face text generation pipeline
generator = pipeline("text-generation", model="distilgpt2")

#Wrap the Hugging Face model pipeline in LangChain
llm = HuggingFacePipeline(pipeline=generator)

#Set up a LangChain prompt template
template = "Explain the impact of AI in job markets: {input_text}"
prompt = PromptTemplate(template=template, input_variables=["input_text"])

#Create an LLMChain with the prompt and model
chain = LLMChain(llm=llm, prompt=prompt)

#Define an input for the prompt and run the chain
input_text = "What is the future of work with AI?"
response = chain.run(input_text=input_text)

#Output the result
print("Generated Response:", response)
