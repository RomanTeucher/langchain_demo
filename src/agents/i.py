from langchain import OpenAI, LLMChain
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool

import src._dotenv
from src.util.vector_store import SomeVectorStore


def load_retrieval_qa() -> RetrievalQA:
    loader = TextLoader("../data/stock.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = SomeVectorStore.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff",
                                     retriever=docsearch.as_retriever())
    return qa


def load_company_naming_tool() -> LLMChain:
    template = """
    You are a naming consultant for new companies.
    What is a good name for a company that makes {product}?
    """

    prompt_template = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.0)

    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


llm = OpenAI(temperature=0.0)

tools = load_tools(["llm-math", "wikipedia"], llm=llm)

tools += [
    Tool(
        name="StockSearch",
        func=load_retrieval_qa().run,
        description="useful for when you want to know about how many and which color socks are in stock",
    ),
    Tool(
        name="ProductNaming",
        func=load_company_naming_tool().run,
        description="useful for when you want to find a clever company name based on a product",
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True)

question = "Which brilliant mind invented socks?"
# question = "Are there red socks in stock?"
# question = "Are there blue  socks in stock?"
# question = "Hey, I produce gloves, what would be a clever company name?"
# question = "Hi, how are you?"
print(agent(question))
