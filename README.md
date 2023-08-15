# Langchain Demo
In this repo there are some sample files per component that show how to use langchain.
Examples are only just that, examples, much more is possible.

# Running Scripts
In order to run scripts you need to provide an open ai key. For that, in src._dotenv.py the .env file (that should be on the top level of that repo) will be loaded. You need to provide .env file in your repo.

# Notes
* The VectorStore is mocked here. The reason is that on Windows you need admin rights to install hnsw lib, so I had to mock it. If you can run regular in memory vector stores, you can just use them.
* I did not show example src.chains.g.py in the demo due to time constraints. That uses routing chains and shows their complexity

