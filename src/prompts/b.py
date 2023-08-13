# Prompts and OutputParsers

import openai
from langchain.prompts import ChatPromptTemplate
import src._dotenv

template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt_template = ChatPromptTemplate.from_template(template)
print(prompt_template)
print("###")
print(prompt_template.format(product="colorful socks"))
print("###")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=[{"role": "user", "content": prompt_template.format(product="colorful socks")}]
)

print(response.choices[0].message["content"])
