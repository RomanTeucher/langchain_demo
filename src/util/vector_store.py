from typing import List, Optional, Any, Iterable, Dict

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VST
from typing_extensions import Type
import openai


class SomeVectorStore(VectorStore):

    def __init__(self, texts: List[str]):
        self.__documents = self.__load(texts)

    # def __load(self, texts) -> Dict[Document, List[float]]:
    #     embeddings = {}
    #     for text in texts:
    #         embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
    #         embeddings[Document(page_content=text)] = embedding
    #     return embeddings

    def __load(self, texts) -> List[Document]:
        return [Document(page_content=text) for text in texts]

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        pass

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return self.__documents

    @classmethod
    def from_texts(cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None,
                   **kwargs: Any) -> VST:
        return SomeVectorStore(texts)
