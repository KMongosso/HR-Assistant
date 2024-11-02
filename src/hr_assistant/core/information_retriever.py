""" Information retriever core class """

from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class InformationRetriever:
    """
    A class to retrieve information from a set of documents using a
    language model and embeddings.

    Attributes:
        retriever (RetrievalQA): An instance of the RetrievalQA chain
            used for information retrieval.

    Methods:
        retrieve_information(query: str) -> str:
            Retrieves relevant information for a given query.
    """

    def __init__(
        self, llm_model_name: str, documents: List[Dict[str, Any]], embedding_path: str
    ) -> None:
        """
        Initializes the InformationRetriever with the specified
        language model and document embeddings.

        Args:
            llm_model_name (str): The name of the language model to use.
            documents (List[Dict[str, Any]]): The list of documents to index for retrieval.
            embedding_path (str): The directory path for saving/loading document embeddings.
        """
        self.retriever = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=llm_model_name),
            retriever=Chroma.from_documents(
                documents=documents,
                embedding=OpenAIEmbeddings(),
                persist_directory=embedding_path,
            ).as_retriever(),
            return_source_documents=True,
        )

    def retrieve_information(self, query: str) -> str:
        """
        Retrieves relevant information based on the given query.

        Args:
            query (str): The query string for information retrieval.

        Returns:
            str: The retrieved information relevant to the query.
        """
        res = self.retriever.invoke(query)
        return res["result"]
