# Chains

import src._dotenv

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

from src.util.vector_store import SomeVectorStore

loader = TextLoader("../data/stock.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embedding_engine = OpenAIEmbeddings()
docsearch = SomeVectorStore.from_documents(texts, embedding_engine)

qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=docsearch.as_retriever())

print(f'Red: {qa.run("Do you have socks in red?")}')
print(f'Blue: {qa.run("Do you have socks in blue?")}')
print(f'Gray: {qa.run("Do you have socks in gray?")}')
print(f'ChitChat: {qa.run("Hey there, what can you do for me?")}')
