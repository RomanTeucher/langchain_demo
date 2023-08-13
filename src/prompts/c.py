# Prompts and OutputParsers

import openai
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import src._dotenv

template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(template)

name_schema = ResponseSchema(name="name_value",
                             description="Extract only the name of the company, no explanatory sentences \
                                    and output it as a json member called 'name'."
                             )
output_parser = StructuredOutputParser.from_response_schemas([name_schema])
format_instructions = output_parser.get_format_instructions()

print(prompt_template.format(product="colorful socks", format_instructions=format_instructions))
print("###")
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=[{"role": "user",
               "content": prompt_template.format(product="colorful socks", format_instructions=format_instructions)}],
)

print(response.choices[0].message["content"])
