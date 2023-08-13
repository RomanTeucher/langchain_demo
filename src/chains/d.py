# Chains


from langchain import  LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import src._dotenv

template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt_template = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0.9)

chain = LLMChain(llm=llm, prompt=prompt_template)

print(chain.run("colorful socks"))



