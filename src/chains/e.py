# Chains


from langchain import LLMChain
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import src._dotenv

product_name_template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

product_name_prompt_template = ChatPromptTemplate.from_template(product_name_template)
llm = ChatOpenAI(temperature=0.0)

product_name_chain = LLMChain(llm=llm, prompt=product_name_prompt_template, output_key="company_name")

translation_template = """
Please translate the product name '{company_name}' to German and French.
"""

translation_prompt_template = ChatPromptTemplate.from_template(translation_template)

translation_chain = LLMChain(llm=llm, prompt=translation_prompt_template, output_key="translation")

overall_chain = SequentialChain(
    chains=[product_name_chain, translation_chain],
    input_variables=["product"],
    output_variables=["company_name", "translation"],
    verbose=True
)
result = overall_chain("colorful socks")
print(result)
