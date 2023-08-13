# Prompts and OutputParsers

import openai
from langchain import PromptTemplate
import src._dotenv

template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt_template = PromptTemplate.from_template(template)
print(prompt_template)
print("###")
print(prompt_template.format(product="colorful socks"))
print("###")

response = openai.Completion.create(
    model="text-davinci-003",
    temperature=0,
    prompt=prompt_template.format(product="colorful socks")
)
print(response.choices[0].text)
